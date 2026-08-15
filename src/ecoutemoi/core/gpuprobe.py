"""Y a-t-il un GPU utilisable sur cette machine ? — sondage sans modèle.

Écoute Moi 2.0 a deux moteurs et deux FORMATS de modèle : ggml pour whisper.cpp
(GPU Vulkan/Metal), CTranslate2 pour faster-whisper (CPU). En backend « auto »,
il faut trancher AVANT de télécharger — sinon on fait payer les deux formats à
tout le monde (665 Mo pour `small` au lieu de 181 ou 484).

Le sondage n'ouvre aucun modèle : `whisper_print_system_info()` suffit à
déclencher l'énumération des backends ggml, qui imprime une ligne
« ggml_vulkan: N = … » par périphérique. Coût : quelques dizaines de
millisecondes.

Il tourne dans un SOUS-PROCESSUS et le résultat est mis en cache sur disque.
Les deux raisons sont les mêmes que pour `engine_proc` : créer une instance
Vulkan exécute le pilote de l'hôte dans notre processus, et un ICD cassé fait
tomber l'application entière — au démarrage, avant même que l'opérateur ait vu
la fenêtre. Un enfant qui meurt, lui, se lit comme « pas de GPU ».

Ce que le sondage prouve : des périphériques sont ÉNUMÉRÉS. Ce qu'il ne prouve
pas : que le backend s'initialisera vraiment (index invalide, mémoire
insuffisante). `core/engines.py` garde donc son repli au chargement réel.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import platformdirs

from ecoutemoi.constants import APP_NAME

log = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 60.0  # extraction PyInstaller onefile comprise
_VULKAN_DEV_RE = re.compile(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\|")
_METAL_RE = re.compile(r"ggml_metal|GPU name:")


@dataclass
class ProbeResult:
    """Ce que la machine a répondu. `gpu` = au moins un candidat crédible."""

    gpu: bool = False
    devices: list[tuple[int, str]] = field(default_factory=list)
    backend_libs: list[str] = field(default_factory=list)
    reason: str | None = None  # pourquoi pas de GPU (message opérateur)
    version: str = ""
    platform: str = ""

    @property
    def summary(self) -> str:
        if not self.gpu:
            return self.reason or "aucun périphérique GPU énuméré"
        if self.devices:
            return " ; ".join(f"{i} — {n}" for i, n in self.devices)
        return "GPU disponible"


def cache_path() -> Path:
    return platformdirs.user_data_path(APP_NAME, appauthor=False) / "gpu_probe.json"


def _fingerprint() -> tuple[str, str]:
    from ecoutemoi import __version__

    return __version__, f"{platform.platform()} ({platform.machine()})"


def load_cache() -> ProbeResult | None:
    """Résultat mémorisé, s'il correspond à CETTE version et à CETTE machine."""
    version, plat = _fingerprint()
    try:
        raw = json.loads(cache_path().read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if raw.get("version") != version or raw.get("platform") != plat:
        return None  # mise à jour de l'app ou de l'OS : on re-sonde
    try:
        return ProbeResult(
            gpu=bool(raw.get("gpu", False)),
            devices=[(int(i), str(n)) for i, n in raw.get("devices", ())],
            backend_libs=[str(x) for x in raw.get("backend_libs", ())],
            reason=raw.get("reason"),
            version=version,
            platform=plat,
        )
    except TypeError, ValueError:
        return None


def save_cache(result: ProbeResult) -> None:
    from ecoutemoi.config import atomic_write_text

    try:
        atomic_write_text(cache_path(), json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
    except OSError as exc:  # dossier en lecture seule : on re-sondera, tant pis
        log.warning("Sondage GPU non mémorisé (%s) : %s", cache_path(), exc)


def clear_cache() -> bool:
    try:
        cache_path().unlink()
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------ parent
def _worker_cmd() -> list[str]:
    if getattr(sys, "frozen", False):  # bundle PyInstaller
        return [sys.executable]
    return [sys.executable, "-m", "ecoutemoi"]


def probe(*, timeout: float = PROBE_TIMEOUT_S) -> ProbeResult:
    """Sonde la machine (sous-processus), SANS toucher au cache."""
    version, plat = _fingerprint()
    result = ProbeResult(version=version, platform=plat)
    with tempfile.TemporaryDirectory(prefix="ecoutemoi-gpuprobe-") as tmp:
        out_path = Path(tmp) / "probe.json"
        cmd = [*_worker_cmd(), "--gpu-probe", str(out_path)]
        try:
            proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        except (subprocess.TimeoutExpired, OSError) as exc:
            result.reason = f"sondage GPU sans réponse ({exc}) — repli CPU"
            log.warning("Sondage GPU en échec : %s", exc)
            return result
        if not out_path.is_file():
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            result.reason = (
                "sondage GPU interrompu (pilote Vulkan instable ?) — repli CPU"
                if proc.returncode != 0
                else "sondage GPU sans résultat — repli CPU"
            )
            log.warning("Sondage GPU code %s : %s", proc.returncode, " / ".join(tail))
            return result
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.reason = f"résultat de sondage illisible ({exc}) — repli CPU"
            return result
    result.devices = [(int(i), str(n)) for i, n in data.get("devices", ())]
    result.backend_libs = [str(x) for x in data.get("backend_libs", ())]
    result.gpu = bool(data.get("gpu", False))
    result.reason = data.get("reason")
    log.info("Sondage GPU : %s", result.summary)
    return result


def gpu_candidates(*, force: bool = False, timeout: float = PROBE_TIMEOUT_S) -> ProbeResult:
    """Résultat du cache, ou sondage neuf (puis mémorisé)."""
    if not force:
        cached = load_cache()
        if cached is not None:
            log.debug("Sondage GPU (cache) : %s", cached.summary)
            return cached
    result = probe(timeout=timeout)
    save_cache(result)
    return result


# ------------------------------------------------------------------- enfant
def worker_main(out_path: str) -> int:
    """`ecoutemoi --gpu-probe OUT.json` : énumère les backends ggml et écrit le JSON.

    Entrées/sorties par FICHIER, pas par tube : dans l'app fenêtrée (Windows,
    double-clic) `sys.stdout` peut être None, et les lignes qui nous intéressent
    partent de toute façon sur le stderr NATIF de ggml.
    """
    from ecoutemoi.core.engine import _StderrCapture, gpu_backend_libs
    from ecoutemoi.core.engine_base import _LogSink

    data: dict = {"gpu": False, "devices": [], "backend_libs": [], "reason": None}
    sink = _LogSink()
    try:
        data["backend_libs"] = gpu_backend_libs()
        import _pywhispercpp as pw

        log_set = getattr(pw, "whisper_log_set", None)
        if log_set is not None:
            log_set(lambda _level, text: sink.write(text))
        try:
            with _StderrCapture(sink):
                # Suffit à faire s'enregistrer les backends ggml : l'énumération
                # Vulkan est imprimée à ce moment-là, sans charger de modèle.
                pw.whisper_print_system_info()
        finally:
            if log_set is not None:
                log_set(None)  # plus aucun pointeur natif vers du Python
    except Exception as exc:
        data["reason"] = f"moteur whisper.cpp indisponible ({type(exc).__name__}: {exc})"
        Path(out_path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return 0

    seen: dict[int, str] = {}
    metal = False
    for line in sink.lines:
        m = _VULKAN_DEV_RE.search(line)
        if m:
            seen.setdefault(int(m.group(1)), m.group(2).strip())
        if _METAL_RE.search(line):
            metal = True
    # Metal n'imprime rien à l'enregistrement du backend (il ne parle qu'à la
    # création du contexte) : sur un Mac Apple Silicon, la présence de la lib
    # ggml-metal EST le signal. Le repli au chargement reste en place si le
    # contexte GPU échoue quand même.
    apple_gpu = sys.platform == "darwin" and (
        metal or any("metal" in lib.lower() for lib in data["backend_libs"])
    )
    data["devices"] = sorted(seen.items())
    data["gpu"] = bool(seen) or apple_gpu
    if not data["gpu"]:
        from ecoutemoi.core.engine import gpu_fallback_reason

        data["reason"] = gpu_fallback_reason(list(sink.lines), bool(data["backend_libs"]))
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return 0


__all__ = ["ProbeResult", "cache_path", "clear_cache", "gpu_candidates", "probe", "worker_main"]
