"""WhisperEngine: pywhispercpp wrapper.

Design notes:
- pywhispercpp is imported lazily so the package imports without the wheel.
- whisper.cpp timestamps are in centiseconds; converted to ms at this boundary.
- The API is process-agnostic on purpose (numpy in, dataclasses out): the same
  engine runs in-process or behind the dedicated subprocess (engine_proc).
- Parameter support is introspected against pywhispercpp's schema; unsupported
  fields are dropped and recorded in `dropped_params`.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecoutemoi.constants import (
    TARGET_SR,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_SAMPLES_OVERLAP,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    WARMUP_AUDIO_S,
    WARMUP_MAX_PASSES,
    WARMUP_STABLE_RATIO,
)
from ecoutemoi.core import cpuinfo

log = logging.getLogger(__name__)

# « ggml_vulkan: 0 = Intel(R) Arc(tm) Graphics (MTL) (pilote…) | uma: 1 | … » :
# une ligne PAR périphérique à l'énumération — index + nom (pilote inclus)
# jusqu'au premier « | ». Le nom court (sans pilote) coupe au premier « ( ».
_VULKAN_DEV_RE = re.compile(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\|")
_VULKAN_SHORT_RE = re.compile(r"^(.+?)(?:\s+\(|$)")
_METAL_RE = re.compile(r"GPU name:\s*(.+)$")
# « whisper_backend_init_gpu: using Vulkan0 backend » vient du logger de whisper :
# c'est LE signal « GPU réellement actif » (l'énumération seule ne suffit pas —
# un index invalide énumère puis retombe en CPU), et il survit même si la
# capture fd du stderr natif a raté et que le nom du périphérique est perdu.
_BACKEND_USING_RE = re.compile(r"whisper_backend_init_gpu: using (Vulkan|Metal)(\d*) backend")
_LANG_RE = re.compile(r"auto-detected language:\s*([a-z]{2,3})")


class _StderrCapture:
    """Capture le stderr C (fd 2) pendant le chargement du modèle.

    Les messages de ggml (détection des périphériques Vulkan/Metal, erreurs de
    pilote) partent sur le stderr NATIF : la redirection de pywhispercpp est
    Python-level pour un sink sans fileno() et ne les voit jamais — la seule
    capture fiable est au niveau du descripteur. Dans l'app fenêtrée Windows
    (double-clic), fd 2 n'existe pas au démarrage : on l'INSTALLE alors sur le
    fichier temporaire (dup2 vers un fd invalide est permis), sinon le sink
    reste vide et l'app affiche « CPU » alors que Vulkan est actif. En sortie,
    faute d'original à restaurer, fd 2 est laissé sur devnull (un fd 2 valide
    évite qu'un descripteur recyclé reçoive les prints natifs suivants).
    """

    def __init__(self, sink: _LogSink):
        self._sink = sink
        self._saved: int | None = None
        self._tmp = None

    def __enter__(self):
        import os
        import tempfile

        try:
            self._saved = os.dup(2)
        except OSError:
            self._saved = None  # fd 2 inexistant (app fenêtrée) : rien à restaurer
        try:
            self._tmp = tempfile.TemporaryFile()
            os.dup2(self._tmp.fileno(), 2)
        except OSError:
            if self._saved is not None:
                os.close(self._saved)
                self._saved = None
            if self._tmp is not None:
                self._tmp.close()
                self._tmp = None
        return self

    def __exit__(self, *exc):
        import os

        if self._tmp is None:
            return False
        try:
            if self._saved is not None:
                os.dup2(self._saved, 2)
            else:  # pas de stderr d'origine : fd 2 valide mais muet
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                os.close(devnull)
        finally:
            if self._saved is not None:
                os.close(self._saved)
        try:
            self._tmp.seek(0)
            data = self._tmp.read().decode("utf-8", errors="replace")
        finally:
            self._tmp.close()
        if data:
            with contextlib.suppress(Exception):  # stderr Python peut être fermé
                sys.stderr.write(data)  # ne pas avaler les messages : re-émettre
            self._sink.write(data if data.endswith("\n") else data + "\n")
        return False


def centi_to_ms(t: int | float) -> int:
    """whisper.cpp segment times are in units of 10 ms (centiseconds)."""
    return round(t * 10)


# Le prompt initial de whisper est PLAFONNÉ à la moitié de la fenêtre de texte du
# décodeur (224 tokens sur 448) ; au-delà, whisper tronque — par la gauche, donc
# silencieusement et par le début de la liste. On borne donc nous-mêmes, en
# caractères, avec une marge : ~4 caractères par token en français.
LEXICON_MAX_CHARS = 700


def normalize_lexicon(text: str) -> str:
    """Lexique opérateur -> prompt initial whisper exploitable.

    Le prompt est du TEXTE, pas une liste : whisper le lit comme le début d'une
    transcription. Une énumération séparée par des virgules suffit à biaiser le
    décodeur vers ces graphies. On aplatit les retours à la ligne (l'opérateur
    saisit volontiers un mot par ligne) et on tronque proprement sur une
    frontière de mot plutôt que de laisser whisper couper au milieu.
    """
    parts = (part.strip(" \t,;") for part in text.replace("\n", ",").split(","))
    flat = ", ".join(part for part in parts if part)
    if len(flat) <= LEXICON_MAX_CHARS:
        return flat
    cut = flat[:LEXICON_MAX_CHARS]
    head, sep, _ = cut.rpartition(", ")
    truncated = head if sep else cut
    log.warning(
        "Lexique tronqué à %d caractères (%d fournis) : whisper plafonne le prompt initial.",
        len(truncated), len(flat),
    )  # fmt: skip
    return truncated


def warmup_audio() -> np.ndarray:
    """~1 s de PAROLE réelle, pour le préchauffage du moteur.

    Du silence ne conviendrait pas : avec le VAD interne de whisper.cpp actif,
    une fenêtre muette est écartée AVANT l'encodeur — aucun shader compilé,
    aucun graphe alloué, et le coût du premier vrai décodage reste entier. On
    réutilise donc la voix de référence déjà embarquée pour le benchmark ; à
    défaut, un signal voisé synthétique (harmoniques à 120 Hz, 4 syllabes/s).
    """
    n = int(TARGET_SR * WARMUP_AUDIO_S)
    try:
        import wave

        import ecoutemoi

        path = Path(ecoutemoi.__file__).resolve().parent / "assets" / "calibration" / "calibration_fr.wav"
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (TARGET_SR, 2, 1):
                w.setpos(min(TARGET_SR, max(0, w.getnframes() - n)))  # saute le silence de tête
                x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
                if x.size >= TARGET_SR // 2:
                    return np.ascontiguousarray(x)
    except Exception as exc:
        log.debug("Voix de préchauffage indisponible (%s) — repli synthétique", exc)
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    voiced = sum(np.sin(2.0 * np.pi * 120.0 * k * t) / k for k in (1, 2, 3, 4, 5))
    envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * 4.0 * t))
    return (0.2 * voiced * envelope).astype(np.float32)


def pick_n_threads(requested: int | None = None) -> int:
    """P-cores physiques (CPU hybride) sinon cœurs physiques ; jamais le SMT.
    Une valeur demandée explicitement est honorée telle quelle."""
    if requested is not None and requested > 0:
        return requested
    return cpuinfo.best_n_threads()


def gpu_backend_libs() -> list[str]:
    """Noms des libs backend GPU (ggml-vulkan/metal) livrées à côté du moteur.

    La présence physique du fichier est le seul signal fiable : quand l'ICD de
    l'hôte est inchargeable (libstdc++ trop vieille, pilote absent), la création
    d'instance Vulkan échoue SANS émettre la moindre ligne « ggml_vulkan » — le
    sink ne permet donc pas de distinguer « wheel CPU » de « pilote cassé ».
    Dossiers couverts : racine du moteur (nos wheels + bundle PyInstaller),
    `pywhispercpp.libs/` (auditwheel) et `pywhispercpp/.dylibs/` (delocate).
    """
    try:
        import _pywhispercpp as pw

        root = Path(pw.__file__).resolve().parent
    except Exception:
        return []
    dirs = (root, root / "pywhispercpp.libs", root / "pywhispercpp" / ".dylibs")
    names: set[str] = set()
    for d in dirs:
        for pat in ("*ggml*vulkan*", "*ggml*metal*"):
            names.update(p.name for p in d.glob(pat) if p.is_file())
    return sorted(names)


def gpu_fallback_reason(sink_lines: list[str], backend_shipped: bool) -> str:
    """Pourquoi le GPU est inactif : wheel CPU, ou backend présent sans périphérique."""
    blob = "\n".join(sink_lines).lower()
    if backend_shipped or "vulkan" in blob or "ggml_metal" in blob:
        return (
            "backend GPU présent mais aucun périphérique utilisable "
            "(pilote Vulkan/ICD de l'hôte absent ou inchargeable)"
        )
    return (
        "moteur compilé sans backend GPU (wheel CPU de PyPI) — utilisez "
        "l'artefact CI ou compilez la wheel Vulkan/Metal (scripts/build_wheel)"
    )


@dataclass
class Segment:
    t0_ms: int
    t1_ms: int
    text: str
    no_speech_prob: float | None = None


@dataclass
class EngineParams:
    model_path: Path
    language: str = "fr"  # "auto" => détection sur le premier énoncé
    translate: bool = False
    n_threads: int | None = None
    backend: str = "auto"  # auto | gpu | cpu
    flash_attn: bool = True
    vad_model_path: Path | None = None
    carry_context: bool = False
    gpu_device: int = 0  # index du périphérique GPU (multi-GPU)
    lexicon: str = ""  # noms propres / acronymes du talk (whisper initial_prompt)


def backend_attempts(backend: str, flash_attn: bool, gpu_device: int = 0) -> list[tuple[str, dict]]:
    """Ordered context-param ladder for a requested backend.

    - "cpu": CPU only, never touches the GPU.
    - "gpu" / "auto": GPU first (with then without flash attention), CPU as the
      final safety net so a broken driver never prevents a session from starting.
    Le rung CPU ne porte jamais gpu_device : il doit rester insensible au GPU.
    """
    if backend == "cpu":
        return [("cpu", {"use_gpu": False, "flash_attn": False})]
    attempts = [
        ("full", {"use_gpu": True, "flash_attn": flash_attn, "gpu_device": gpu_device}),
        ("no-flash", {"use_gpu": True, "flash_attn": False, "gpu_device": gpu_device}),
        ("cpu", {"use_gpu": False, "flash_attn": False}),
    ]
    seen: set[tuple] = set()
    out: list[tuple[str, dict]] = []
    for name, ctx in attempts:
        key = tuple(sorted(ctx.items()))
        if key not in seen:
            seen.add(key)
            out.append((name, ctx))
    return out


def supported_context_keys() -> set[str] | None:
    """Clés context_params acceptées par le binding, ou None si inconnues.

    pywhispercpp 1.5.0 REFUSE les clés inconnues (ValueError) : on filtre donc
    en amont par introspection du TypedDict ContextParams plutôt que de laisser
    toute l'échelle GPU échouer sur un binding plus ancien sans `gpu_device`.
    """
    try:
        from pywhispercpp.model import ContextParams

        keys = set(getattr(ContextParams, "__annotations__", ()) or ())
        return keys or None
    except Exception:
        return None


class _LogSink:
    """File-like object capturing whisper.cpp log lines (backend + language info)."""

    def __init__(self, keep: int = 800):
        self.lines: deque[str] = deque(maxlen=keep)
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s: str) -> None:
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    self.lines.append(line)
                    log.debug("whisper.cpp: %s", line)

    def flush(self) -> None:
        pass


class WhisperEngine:
    """One Model instance per (model, backend); a single thread calls transcribe()."""

    def __init__(self, params: EngineParams):
        self.params = params
        self.sink = _LogSink()
        self.n_threads = pick_n_threads(params.n_threads)
        self.vad_active = False
        self.flash_attn_active = False
        self.dropped_params: list[str] = []
        self.load_variant = "?"
        self.load_s = 0.0
        self.last_decode_ms = 0.0
        self.warmup_ms: list[float] = []
        self._gpu_requested = False
        self._model = None
        self._load()

    # ------------------------------------------------------------------ load
    def _full_params(self) -> dict:
        params = {
            "language": self.params.language,
            "translate": self.params.translate,
            "n_threads": self.n_threads,
            "no_context": not self.params.carry_context,
            "suppress_blank": True,
            "temperature": 0.0,
            "temperature_inc": 0.0,
            "no_speech_thold": 0.6,
            "print_progress": False,
            "print_realtime": False,
            "print_timestamps": False,
            "print_special": False,
        }
        lexicon = normalize_lexicon(self.params.lexicon)
        if lexicon:
            # `carry_initial_prompt` est INDISPENSABLE en streaming : sans lui le
            # prompt ne biaise que la première fenêtre décodée, alors qu'on
            # redécode une fenêtre glissante en continu. Avec, le lexique est
            # rappelé au décodeur à chaque passe.
            params["initial_prompt"] = lexicon
            params["carry_initial_prompt"] = True
        return params

    def _vad_params(self) -> dict:
        """pywhispercpp 1.5.0 exposes `vad` + `vad_model_path` but NOT the fine
        `vad_params` struct (threshold/min_speech/min_silence/pad/overlap): those stay
        at whisper.cpp defaults — deviation recorded in dropped_params."""
        if self.params.vad_model_path is None:
            return {}
        return {
            "vad": True,
            "vad_model_path": str(self.params.vad_model_path),
            "vad_params": {
                "threshold": VAD_THRESHOLD,
                "min_speech_duration_ms": VAD_MIN_SPEECH_MS,
                "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
                "speech_pad_ms": VAD_SPEECH_PAD_MS,
                "samples_overlap": VAD_SAMPLES_OVERLAP,
            },
        }

    @staticmethod
    def _schema_keys() -> set[str] | None:
        """Valid pywhispercpp full-params names, or None if the schema is unavailable."""
        try:
            from pywhispercpp.constants import PARAMS_SCHEMA

            return set(PARAMS_SCHEMA.keys())
        except Exception:
            return None

    def _install_whisper_log_callback(self) -> bool:
        """Route les logs whisper.cpp/ggml vers le sink via whisper_log_set.

        whisper_log_set branche AUSSI le logger ggml : les lignes de détection
        Vulkan/Metal (« ggml_vulkan: 0 = … ») arrivent par ce canal même dans
        l'app fenêtrée Windows, où le stderr C est débranché (FILE* sans
        descripteur valide) et où toute capture au niveau fd est vaine — sans
        ce callback, le sink reste vide et l'app affiche « CPU » alors que
        Vulkan est actif.

        DANGER apparié : ce callback global pointe vers du Python. Laissé
        installé au-delà de la vie de l'interpréteur, le teardown natif de
        ggml rappelle du Python pendant Py_Finalize → segfault de SORTIE.
        D'où le reset dans close() ET la ceinture atexit ci-dessous.
        """
        try:
            import _pywhispercpp as pw

            log_set = getattr(pw, "whisper_log_set", None)
            if log_set is None:
                return False
            sink = self.sink

            def _on_log(_level: int, text: str) -> None:
                sink.write(text)

            self._log_callback = _on_log  # référence gardée (callback global côté C)
            log_set(_on_log)
            _register_log_callback_atexit()
            return True
        except Exception:
            return False

    def _load(self) -> None:
        from pywhispercpp.model import Model

        if not self._install_whisper_log_callback():
            log.debug("whisper_log_set indisponible — détection GPU via stderr seul")
        try:
            init_sig = set(inspect.signature(Model.__init__).parameters)
        except TypeError, ValueError:
            init_sig = set()
        init_kwargs: dict = {}
        if "redirect_whispercpp_logs_to" in init_sig:
            init_kwargs["redirect_whispercpp_logs_to"] = self.sink

        wanted = self._full_params() | self._vad_params()
        schema = self._schema_keys()
        if schema is not None:
            params = {k: v for k, v in wanted.items() if k in schema}
            self.dropped_params = sorted(set(wanted) - set(params))
            if self.dropped_params:
                log.warning("pywhispercpp: unsupported params dropped: %s", self.dropped_params)
        else:  # very old/odd binding: pass everything and rely on the ladder
            params = wanted
            self.dropped_params = []

        supports_ctx = "context_params" in init_sig
        ctx_keys = supported_context_keys() if supports_ctx else None
        ctx_warned: set[str] = set()

        def build_attempts(gpu_device: int) -> list[tuple[str, dict, dict]]:
            # Fallback ladder on context params (per the requested backend), then
            # a last attempt without the internal VAD (if its model fails to load).
            atts: list[tuple[str, dict, dict]] = [
                (name, ctx, params)
                for name, ctx in backend_attempts(self.params.backend, self.params.flash_attn, gpu_device)
            ]
            no_vad = {k: v for k, v in params.items() if not k.startswith("vad")}
            if no_vad != params:
                atts.append(("cpu-no-vad", {"use_gpu": False, "flash_attn": False}, no_vad))
            return atts

        def run_ladder(attempts: list[tuple[str, dict, dict]]) -> Exception | None:
            last: Exception | None = None
            for name, ctx, kw in attempts:
                if ctx_keys is not None:  # binding plus ancien : clés inconnues refusées
                    unknown = set(ctx) - ctx_keys
                    for k in sorted(unknown - ctx_warned):
                        log.warning("context_params.%s non supporté par ce binding — ignoré", k)
                    ctx_warned.update(unknown)
                    ctx = {k: v for k, v in ctx.items() if k in ctx_keys}
                try:
                    t0 = time.perf_counter()
                    extra = {"context_params": ctx} if supports_ctx else {}
                    with _StderrCapture(self.sink):
                        self._model = Model(str(self.params.model_path), **init_kwargs, **extra, **kw)
                    self.load_s = time.perf_counter() - t0
                    self.load_variant = name
                    self.vad_active = bool(kw.get("vad", False))
                    self.flash_attn_active = bool(ctx.get("flash_attn", False)) and supports_ctx
                    # use_gpu réellement passé au contexte chargé : la lib Vulkan énumère
                    # les périphériques même en CPU forcé, le sink ne suffit donc pas.
                    self._gpu_requested = bool(ctx.get("use_gpu", False)) if supports_ctx else True
                    log.info(
                        "Engine loaded (%s) in %.1f s: %s [threads=%d, vad=%s, flash_attn=%s, gpu_device=%s]",
                        name, self.load_s, Path(self.params.model_path).name,
                        self.n_threads, self.vad_active, self.flash_attn_active,
                        ctx.get("gpu_device", "-"),
                    )  # fmt: skip
                    return None
                except Exception as exc:
                    last = exc
                    log.warning("Engine load attempt '%s' failed: %s", name, exc)
            return last

        last_exc = run_ladder(build_attempts(self.params.gpu_device))
        if self._model is None:
            raise RuntimeError(f"Impossible de charger le modèle whisper: {last_exc}") from last_exc

        # Index GPU hors limites : whisper retombe en CPU SANS lever d'erreur —
        # l'échelle ne peut pas l'attraper. Un seul rechargement correctif sur le
        # périphérique 0 pour qu'un settings.json erroné ne coûte pas le GPU.
        devices = self.gpu_devices()
        if (
            self._gpu_requested
            and not self.gpu_active()
            and devices
            and self.params.gpu_device >= len(devices)
        ):
            log.warning(
                "Index GPU %d invalide (%d périphérique(s)) — rechargement sur le GPU 0.",
                self.params.gpu_device, len(devices),
            )  # fmt: skip
            self._model = None
            self.params.gpu_device = 0
            last_exc = run_ladder(build_attempts(0))
            if self._model is None:
                raise RuntimeError(f"Impossible de charger le modèle whisper: {last_exc}") from last_exc

        if devices:
            log.info("Périphériques Vulkan : %s", " ; ".join(f"{i} — {n}" for i, n in devices))
        if self.params.backend == "gpu" and not self.gpu_active():
            log.warning("Backend GPU demandé mais aucun GPU utilisable — repli CPU.")

    # ------------------------------------------------------------- transcribe
    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        """Decode one float32 mono 16 kHz window; returns segments in ms."""
        assert self._model is not None
        t0 = time.perf_counter()
        raw = self._model.transcribe(audio)
        self.last_decode_ms = (time.perf_counter() - t0) * 1000.0
        out: list[Segment] = []
        for s in raw:
            text = (s.text or "").strip()
            if not text:
                continue
            nsp = getattr(s, "no_speech_prob", None)
            out.append(Segment(centi_to_ms(s.t0), centi_to_ms(s.t1), text, nsp))
        return out

    # ---------------------------------------------------------------- préchauffage
    def warmup(self, on_pass=None, should_stop=None) -> list[float]:
        """Décodages à blanc jusqu'à stabilisation ; retourne les temps (ms).

        Le premier décodage d'un moteur frais paie la compilation des shaders
        Vulkan, l'allocation du graphe et le remplissage des caches : de quelques
        secondes à une minute au tout premier lancement sur une machine donnée.
        Payé PENDANT la session, ce coût produit un tampon d'audio en retard puis
        une rafale de rattrapage avant de revenir au temps réel. Payé ici, avant
        l'ouverture du micro, il ne coûte que de l'attente au démarrage.

        `on_pass(i, total)` suit la progression, `should_stop()` interrompt.
        """
        audio = warmup_audio()
        times: list[float] = []
        for i in range(1, WARMUP_MAX_PASSES + 1):
            if should_stop is not None and should_stop():
                break
            if on_pass is not None:
                on_pass(i, WARMUP_MAX_PASSES)
            try:
                self.transcribe(audio)
            except Exception as exc:  # un échec ici se reproduira en session
                log.warning("Passe de préchauffage %d en échec : %s", i, exc)
                break
            times.append(self.last_decode_ms)
            # Stable dès qu'une passe retombe au niveau de la meilleure observée :
            # le surcoût unique (shaders, caches) est absorbé.
            if len(times) >= 2 and times[-1] <= min(times) * WARMUP_STABLE_RATIO:
                break
        self.warmup_ms = times
        if times:
            log.info(
                "Préchauffage : %d passe(s) — %s ms",
                len(times), ", ".join(f"{t:.0f}" for t in times),
            )  # fmt: skip
        return times

    # ---------------------------------------------------------------- helpers
    def detected_language(self) -> str | None:
        for line in reversed(self.sink.lines):
            m = _LANG_RE.search(line)
            if m:
                return m.group(1)
        # Fallback: low-level whisper_full_lang_id on the model context.
        try:
            import _pywhispercpp as pw

            ctx = getattr(self._model, "_ctx", None)
            if ctx is not None:
                lang_id = pw.whisper_full_lang_id(ctx)
                if lang_id >= 0:
                    return pw.whisper_lang_str(lang_id)
        except Exception:
            pass
        return None

    def gpu_devices(self) -> list[tuple[int, str]]:
        """Périphériques Vulkan énumérés au chargement : [(index, « nom (pilote) »)].

        L'énumération est imprimée par ggml même quand le GPU n'est finalement
        pas utilisé — c'est une liste de CANDIDATS, pas le périphérique actif.
        """
        seen: dict[int, str] = {}
        for line in self.sink.lines:
            m = _VULKAN_DEV_RE.search(line)
            if m:
                seen.setdefault(int(m.group(1)), m.group(2).strip())
        return sorted(seen.items())

    def active_gpu_index(self) -> int | None:
        """Index du périphérique GPU réellement utilisé (None si CPU)."""
        for line in reversed(self.sink.lines):
            m = _BACKEND_USING_RE.search(line)
            if m:
                return int(m.group(2)) if m.group(2) else 0
        return None

    def backend_info(self) -> str:
        if getattr(self, "_gpu_requested", True):
            api: str | None = None
            idx = 0
            for line in self.sink.lines:
                m = _METAL_RE.search(line)
                if m:
                    return f"Metal : {m.group(1).strip()}"
                m = _BACKEND_USING_RE.search(line)
                if m:
                    api = m.group(1)
                    idx = int(m.group(2)) if m.group(2) else 0
            if api == "Metal":
                return "Metal : GPU"
            if api:  # Vulkan actif : nom court du périphérique choisi
                name = dict(self.gpu_devices()).get(idx)
                if name:
                    short = _VULKAN_SHORT_RE.match(name)
                    return f"Vulkan : {short.group(1).strip() if short else name}"
                return f"Vulkan : GPU {idx}"
        return f"CPU ({self.n_threads} threads)"

    def gpu_active(self) -> bool:
        """True when whisper.cpp actually reported a Vulkan/Metal device."""
        return not self.backend_info().startswith("CPU")

    def gpu_diagnostic(self) -> str | None:
        """Pourquoi le GPU n'est pas actif (None s'il l'est).

        Distingue les causes concrètes :
        - index GPU demandé hors limites (liste alors les périphériques valides) ;
        - périphériques détectés mais backend non initialisé (voir le log) ;
        - moteur sans backend GPU compilé (wheel PyPI = CPU pur) ;
        - backend présent mais pilote/ICD de l'hôte absent ou inchargeable.
        """
        if self.gpu_active():
            return None
        devices = self.gpu_devices()
        requested = getattr(self.params, "gpu_device", 0)
        if devices:
            listing = " ; ".join(f"{i} — {n}" for i, n in devices)
            if requested >= len(devices):
                return f"index GPU {requested} invalide — périphériques disponibles : {listing}"
            return (
                "périphériques Vulkan détectés mais backend non initialisé "
                f"(voir le fichier de log) : {listing}"
            )
        return gpu_fallback_reason(list(self.sink.lines), bool(gpu_backend_libs()))

    def diagnostics(self) -> dict:
        """État structuré du moteur pour --diag et le dialogue GUI « Diagnostic »."""
        info: dict = {
            "modele": Path(self.params.model_path).name,
            "backend": self.backend_info(),
            "gpu_actif": self.gpu_active(),
            "backend_demande": self.params.backend,
            "gpu_device_demande": self.params.gpu_device,
            "gpu_device_actif": self.active_gpu_index(),
            "gpu_devices": [f"{i} — {n}" for i, n in self.gpu_devices()],
            "libs_backend_gpu": gpu_backend_libs(),
            "variante_chargement": self.load_variant,
            "chargement_s": round(self.load_s, 2),
            "prechauffage_ms": [round(t) for t in self.warmup_ms],
            "flash_attn": self.flash_attn_active,
            "vad": self.vad_active,
            "threads": self.n_threads,
            "params_ignores": self.dropped_params,
            "lexique_caracteres": len(normalize_lexicon(self.params.lexicon)),
            "diagnostic_gpu": self.gpu_diagnostic(),
        }
        try:
            import _pywhispercpp as pw

            info["system_info"] = pw.whisper_print_system_info()
        except Exception:
            pass
        return info

    def close(self) -> None:
        self._model = None
        if getattr(self, "_log_callback", None) is not None:
            _reset_whisper_log_callback()
            self._log_callback = None


_atexit_registered = False


def _reset_whisper_log_callback() -> None:
    """whisper/ggml reprennent leur log par défaut (stderr) — plus aucun
    pointeur natif vers du Python."""
    with contextlib.suppress(Exception):
        import _pywhispercpp as pw

        pw.whisper_log_set(None)


def _register_log_callback_atexit() -> None:
    """Ceinture : même si close() n'est jamais appelé (CLI interrompue, crash
    applicatif), le callback est désinstallé AVANT la finalisation de
    l'interpréteur — sinon segfault de sortie garanti côté natif."""
    global _atexit_registered
    if not _atexit_registered:
        import atexit

        atexit.register(_reset_whisper_log_callback)
        _atexit_registered = True
