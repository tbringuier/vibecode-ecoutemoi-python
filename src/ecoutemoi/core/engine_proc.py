"""Moteur dans un PROCESSUS séparé, derrière la même API que les moteurs locaux.

L'enfant charge celui des deux moteurs que `core/engines.py` retient
(whisper.cpp au GPU, faster-whisper au CPU) : le parent, lui, ne voit qu'une
seule surface — c'est tout l'intérêt.

Pourquoi : `transcribe()` passe des secondes entières dans du code natif. Dans
le processus Qt, cela se paie deux fois —

1. **Gigue GIL** : le callback audio PortAudio et le rendu Qt sont des threads
   Python. Ils ne reprennent la main qu'aux relâchements du GIL par l'extension,
   d'où des blocs audio perdus et une interface qui saccade pendant le décodage
   (c'est exactement ce que mesure `--measure-gil`).
2. **Fragilité** : un pilote Vulkan qui segfault, un modèle corrompu, un OOM
   natif — et c'est toute l'application qui meurt, en pleine conférence.

Déplacer l'inférence dans un enfant règle les deux : le processus Qt ne fait plus
que des entrées/sorties sur un tube, et la mort de l'enfant est rattrapable —
`SubprocessEngine` le relance et rejoue le chargement.

Protocole : trames `longueur (uint32 big-endian) + pickle` sur stdin/stdout de
l'enfant. L'audio voyage en float32 brut à côté de l'en-tête, sans passer par
pickle (une fenêtre de 9 s = 576 Ko, une copie mémoire, pas de sérialisation).
stdout de l'enfant est réservé au protocole : ses logs partent sur stderr, que le
parent recopie dans le fichier de log.
"""

from __future__ import annotations

import contextlib
import logging
import pickle
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ecoutemoi.core.engine_base import EngineParams, Segment

log = logging.getLogger(__name__)

_HEADER = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 << 20  # 64 Mio : au-delà, le tube est désynchronisé
LOAD_TIMEOUT_S = 600.0  # premier lancement : téléchargement de shaders/modèle déjà fait, mais CPU lent
CALL_TIMEOUT_S = 300.0  # un décodage qui dépasse 5 min, c'est un enfant mort
MAX_RESTARTS = 3


class EngineProcessError(RuntimeError):
    """L'enfant est mort, ne répond plus, ou a refusé la commande."""


# --------------------------------------------------------------- cadrage
def write_frame(stream, payload: dict, blob: bytes = b"") -> None:
    """Écrit `payload` (pickle) suivi de `blob` (octets bruts)."""
    head = pickle.dumps({**payload, "_blob_len": len(blob)}, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(head)))
    stream.write(head)
    if blob:
        stream.write(blob)
    stream.flush()


def _read_exactly(stream, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("tube fermé")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream) -> tuple[dict, bytes]:
    """Lit une trame écrite par `write_frame`. Lève EOFError si le tube ferme."""
    (size,) = _HEADER.unpack(_read_exactly(stream, _HEADER.size))
    if size > MAX_MESSAGE_BYTES:
        raise EngineProcessError(f"trame de {size} octets refusée (tube désynchronisé ?)")
    payload = pickle.loads(_read_exactly(stream, size))
    blob_len = int(payload.pop("_blob_len", 0))
    if blob_len > MAX_MESSAGE_BYTES:
        raise EngineProcessError(f"blob de {blob_len} octets refusé")
    return payload, (_read_exactly(stream, blob_len) if blob_len else b"")


def _worker_cmd() -> list[str]:
    """Même exécutable que nous : marche sous uv/venv comme dans le bundle."""
    if getattr(sys, "frozen", False):  # PyInstaller
        return [sys.executable]
    return [sys.executable, "-m", "ecoutemoi"]


# ------------------------------------------------------------------ parent
class SubprocessEngine:
    """Façade locale ; l'inférence vit dans l'enfant.

    Surface volontairement identique à `WhisperEngine` : le `Streamer` et la GUI
    ne savent pas lequel des deux ils manipulent. Les valeurs constatées au
    chargement (backend, périphériques GPU, diagnostics) sont rapatriées une fois
    et servies depuis le cache — pas de va-et-vient par appel.
    """

    def __init__(self, params: EngineParams):
        self.params = params
        self.n_threads = 0
        self.vad_active = False
        self.flash_attn_active = False
        self.dropped_params: list[str] = []
        self.load_variant = "?"
        self.load_s = 0.0
        self.last_decode_ms = 0.0
        self.warmup_ms: list[float] = []
        self.restarts = 0
        self.name = "?"  # moteur réellement chargé par l'enfant
        self.fallback_reason: str | None = None
        self._backend_info = "CPU"
        self._gpu_devices: list[tuple[int, str]] = []
        self._gpu_active = False
        self._gpu_diagnostic: str | None = None
        self._active_gpu_index: int | None = None
        self._diagnostics: dict = {}
        self._detected_language: str | None = None
        self._sink_lines: list[str] = []
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._start()

    # -- cycle de vie de l'enfant
    def _start(self) -> None:
        cmd = [*_worker_cmd(), "--engine-worker"]
        log.info("Démarrage du moteur en sous-processus : %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc.stderr,), daemon=True, name="engine-stderr"
        )
        self._stderr_thread.start()
        info = self._call("load", {"params": _params_to_dict(self.params)}, timeout=LOAD_TIMEOUT_S)
        self._absorb_load(info)

    @staticmethod
    def _drain_stderr(stream) -> None:
        """Le stderr de l'enfant (logs whisper.cpp compris) rejoint notre log."""
        with contextlib.suppress(Exception):
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    log.debug("moteur: %s", line)

    def _absorb_load(self, info: dict) -> None:
        self.name = str(info.get("engine_name", "?"))
        self.fallback_reason = info.get("fallback_reason")
        self.n_threads = int(info.get("n_threads", 0))
        self.vad_active = bool(info.get("vad_active", False))
        self.flash_attn_active = bool(info.get("flash_attn_active", False))
        self.dropped_params = list(info.get("dropped_params", ()))
        self.load_variant = str(info.get("load_variant", "?"))
        self.load_s = float(info.get("load_s", 0.0))
        self._backend_info = str(info.get("backend_info", "CPU"))
        self._gpu_devices = [(int(i), str(n)) for i, n in info.get("gpu_devices", ())]
        self._gpu_active = bool(info.get("gpu_active", False))
        self._gpu_diagnostic = info.get("gpu_diagnostic")
        self._active_gpu_index = info.get("active_gpu_index")
        self._diagnostics = dict(info.get("diagnostics", {}))
        self._sink_lines = list(info.get("sink_lines", ()))
        # L'index GPU peut avoir été corrigé côté enfant (index hors limites).
        with contextlib.suppress(Exception):
            self.params.gpu_device = int(info.get("gpu_device", self.params.gpu_device))

    def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            with contextlib.suppress(Exception):
                stream.close()
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                proc.kill()

    def _restart(self, reason: str) -> None:
        """Relance l'enfant et rejoue le chargement. Lève après MAX_RESTARTS."""
        if self.restarts >= MAX_RESTARTS:
            raise EngineProcessError(
                f"moteur mort {self.restarts + 1} fois ({reason}) — abandon. "
                "Voir le fichier de log ; le mode « moteur dans le processus » "
                "(Réglages avancés) contourne le problème au prix de la stabilité."
            )
        self.restarts += 1
        log.warning("Moteur relancé (tentative %d/%d) après : %s", self.restarts, MAX_RESTARTS, reason)
        self._terminate()
        self._start()

    # -- protocole
    def _call(self, cmd: str, payload: dict | None = None, *, blob: bytes = b"",
              timeout: float = CALL_TIMEOUT_S) -> dict:  # fmt: skip
        """Un aller-retour avec l'enfant. Le verrou sérialise les appelants."""
        with self._lock:
            if self._closed:
                raise EngineProcessError("moteur déjà fermé")
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise EngineProcessError("sous-processus moteur absent")
            deadline = time.monotonic() + timeout
            try:
                write_frame(proc.stdin, {"cmd": cmd, **(payload or {})}, blob)
                reply, _ = read_frame(proc.stdout)
            except (EOFError, BrokenPipeError, OSError) as exc:
                code = proc.poll()
                raise EngineProcessError(
                    f"sous-processus moteur mort pendant « {cmd} » (code {code}) : {exc}"
                ) from exc
            if time.monotonic() > deadline:
                log.warning("Réponse « %s » arrivée après le délai de %.0f s", cmd, timeout)
            if not reply.get("ok"):
                raise EngineProcessError(f"« {cmd} » a échoué côté moteur : {reply.get('error')}")
            return reply

    def _call_resilient(self, cmd: str, payload: dict | None = None, *, blob: bytes = b"") -> dict:
        """Comme `_call`, mais relance l'enfant une fois s'il est mort."""
        try:
            return self._call(cmd, payload, blob=blob)
        except EngineProcessError as exc:
            if self._closed:
                raise
            self._restart(str(exc))
            return self._call(cmd, payload, blob=blob)

    # -- API WhisperEngine
    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        contiguous = np.ascontiguousarray(audio, dtype=np.float32)
        reply = self._call_resilient("transcribe", blob=contiguous.tobytes())
        self.last_decode_ms = float(reply.get("decode_ms", 0.0))
        lang = reply.get("detected_language")
        if lang:
            self._detected_language = lang
        return [Segment(*row) for row in reply.get("segments", ())]

    def warmup(self, on_pass=None, should_stop=None) -> list[float]:
        """Le préchauffage a lieu DANS l'enfant : la progression revient d'un coup.

        Découper le préchauffage en un aller-retour par passe n'apporterait rien
        (les passes sont pilotées par le moteur, pas par nous) et exposerait
        l'appelant à un tube à demi-lu si l'enfant meurt entre deux passes.
        """
        if should_stop is not None and should_stop():
            return []
        reply = self._call_resilient("warmup")
        self.warmup_ms = [float(t) for t in reply.get("warmup_ms", ())]
        self._diagnostics = dict(reply.get("diagnostics", self._diagnostics))
        if on_pass is not None and self.warmup_ms:
            total = len(self.warmup_ms)
            for i in range(1, total + 1):
                on_pass(i, total)
        return self.warmup_ms

    def detected_language(self) -> str | None:
        return self._detected_language

    def gpu_devices(self) -> list[tuple[int, str]]:
        return list(self._gpu_devices)

    def active_gpu_index(self) -> int | None:
        return self._active_gpu_index

    def backend_info(self) -> str:
        return self._backend_info

    def gpu_active(self) -> bool:
        return self._gpu_active

    def gpu_diagnostic(self) -> str | None:
        return self._gpu_diagnostic

    def diagnostics(self) -> dict:
        info = dict(self._diagnostics)
        info["processus"] = "sous-processus dédié"
        info.setdefault("moteur", self.name)
        info["prechauffage_ms"] = [round(t) for t in self.warmup_ms]
        if self.restarts:
            info["redemarrages_moteur"] = self.restarts
        return info

    @property
    def sink(self):
        """Compat `engine.sink.lines` (rapport de diagnostic)."""
        return _SinkView(self._sink_lines)

    def close(self) -> None:
        self._closed = True
        with contextlib.suppress(Exception):
            if self._proc is not None and self._proc.poll() is None:
                with self._lock:
                    write_frame(self._proc.stdin, {"cmd": "close"})
        self._terminate()


class _SinkView:
    """Vue en lecture seule des lignes whisper.cpp remontées de l'enfant."""

    def __init__(self, lines: list[str]):
        self.lines = list(lines)


def _params_to_dict(params: EngineParams) -> dict:
    return {
        "model_path": str(params.model_path) if params.model_path else None,
        "language": params.language,
        "translate": bool(params.translate),
        "n_threads": params.n_threads,
        "backend": params.backend,
        "flash_attn": bool(params.flash_attn),
        "vad_model_path": str(params.vad_model_path) if params.vad_model_path else None,
        "carry_context": bool(params.carry_context),
        "gpu_device": int(params.gpu_device),
        "lexicon": params.lexicon,
        "ct2_path": str(params.ct2_path) if params.ct2_path else None,
        "compute_type": params.compute_type,
        "beam_size": int(params.beam_size),
        "engine": params.engine,
        "vad_filter": bool(params.vad_filter),
    }


def _params_from_dict(data: dict) -> EngineParams:
    return EngineParams(
        model_path=Path(data["model_path"]) if data.get("model_path") else None,
        language=data.get("language", "fr"),
        translate=bool(data.get("translate", False)),
        n_threads=data.get("n_threads"),
        backend=data.get("backend", "auto"),
        flash_attn=bool(data.get("flash_attn", True)),
        vad_model_path=Path(data["vad_model_path"]) if data.get("vad_model_path") else None,
        carry_context=bool(data.get("carry_context", False)),
        gpu_device=int(data.get("gpu_device", 0)),
        lexicon=data.get("lexicon", "") or "",
        ct2_path=Path(data["ct2_path"]) if data.get("ct2_path") else None,
        compute_type=data.get("compute_type", "int8"),
        beam_size=max(1, int(data.get("beam_size", 1))),
        engine=data.get("engine", "auto"),
        vad_filter=bool(data.get("vad_filter", True)),
    )


# ------------------------------------------------------------------- enfant
def worker_main() -> int:
    """Boucle du sous-processus moteur (`ecoutemoi --engine-worker`).

    stdout appartient au protocole : on le détache tout de suite pour que rien
    (print d'une dépendance, avertissement) ne puisse corrompre le cadrage.
    """
    import os

    from ecoutemoi.core.engines import create_engine

    stdin_raw = sys.stdin.buffer if sys.stdin is not None else open(os.devnull, "rb")  # noqa: SIM115
    if sys.stdout is None:  # app fenêtrée : pas de stdout, rien à faire ici
        return 1
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    sys.stdout = sys.stderr  # le tube de protocole ne doit recevoir aucun print applicatif

    engine = None
    try:
        while True:
            try:
                request, blob = read_frame(stdin_raw)
            except EOFError:
                return 0  # parent parti
            cmd = request.get("cmd")
            try:
                if cmd == "load":
                    engine = create_engine(_params_from_dict(request["params"]))
                    write_frame(protocol_out, {"ok": True, **_load_info(engine)})
                elif cmd == "transcribe":
                    if engine is None:
                        raise RuntimeError("transcribe avant load")
                    audio = np.frombuffer(blob, dtype=np.float32)
                    segments = engine.transcribe(audio)
                    write_frame(
                        protocol_out,
                        {
                            "ok": True,
                            "decode_ms": engine.last_decode_ms,
                            "detected_language": engine.detected_language(),
                            "segments": [(s.t0_ms, s.t1_ms, s.text, s.no_speech_prob) for s in segments],
                        },
                    )
                elif cmd == "warmup":
                    if engine is None:
                        raise RuntimeError("warmup avant load")
                    times = engine.warmup()
                    write_frame(
                        protocol_out,
                        {"ok": True, "warmup_ms": times, "diagnostics": engine.diagnostics()},
                    )
                elif cmd == "close":
                    return 0
                else:
                    raise RuntimeError(f"commande inconnue : {cmd!r}")
            except Exception as exc:
                log.exception("Commande moteur « %s » en échec", cmd)
                # Répondre l'erreur AU LIEU de mourir : le parent la remonte à
                # l'opérateur, et une commande fautive ne coûte pas la session.
                write_frame(protocol_out, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.close()
        with contextlib.suppress(Exception):
            protocol_out.close()


def _load_info(engine) -> dict:
    return {
        "engine_name": getattr(engine, "name", "?"),
        "fallback_reason": getattr(engine, "fallback_reason", None),
        "n_threads": engine.n_threads,
        "vad_active": engine.vad_active,
        "flash_attn_active": engine.flash_attn_active,
        "dropped_params": engine.dropped_params,
        "load_variant": engine.load_variant,
        "load_s": engine.load_s,
        "backend_info": engine.backend_info(),
        "gpu_devices": engine.gpu_devices(),
        "gpu_active": engine.gpu_active(),
        "gpu_diagnostic": engine.gpu_diagnostic(),
        "active_gpu_index": engine.active_gpu_index(),
        "gpu_device": engine.params.gpu_device,
        "diagnostics": engine.diagnostics(),
        "sink_lines": list(engine.sink.lines),
    }


__all__ = ["EngineProcessError", "SubprocessEngine", "read_frame", "worker_main", "write_frame"]
