"""Quel moteur pour quelle machine — la seule fabrique de moteurs.

La règle tient en deux lignes :

    GPU (Vulkan / Metal)  -> whisper.cpp   (`core/engine.py`)
    CPU                   -> faster-whisper (`core/engine_fw.py`)

parce que chacun est le meilleur là où l'autre ne va pas. CTranslate2, sous
faster-whisper, ne connaît que le CPU : sur un GPU, quel qu'en soit le vendeur,
il n'existe tout simplement pas. Inversement, sur CPU, whisper.cpp se fait
distancer d'un facteur 3 sur `small`. Aucun des deux ne couvre le terrain seul.

`backend` (réglage opérateur) décide :
- « cpu »  : faster-whisper, sans jamais toucher au GPU ;
- « gpu »  : whisper.cpp, avec son échelle de repli interne ;
- « auto » : whisper.cpp sur GPU si un périphérique répond VRAIMENT, sinon
  faster-whisper.

Chaque repli est explicite et journalisé : personne ne doit découvrir en pleine
conférence qu'il tourne sur un moteur qu'il n'a pas choisi.
"""

from __future__ import annotations

import dataclasses
import logging

from ecoutemoi.constants import ENGINE_FASTER_WHISPER, ENGINE_WHISPERCPP
from ecoutemoi.core.engine_base import EngineParams

log = logging.getLogger(__name__)


def engine_for_backend(backend: str, cpu_engine: str = ENGINE_FASTER_WHISPER) -> str:
    """Moteur impliqué par un backend demandé (avant toute vérification matérielle)."""
    if backend == "cpu":
        return cpu_engine
    if backend == "gpu":
        return ENGINE_WHISPERCPP
    return "auto"


def _faster_whisper_ready(params: EngineParams) -> tuple[bool, str | None]:
    """(utilisable ?, raison de l'indisponibilité)."""
    from ecoutemoi.core import engine_fw

    if params.ct2_path is None:
        return False, "aucun modèle CTranslate2 installé pour ce modèle"
    if not engine_fw.available():
        return False, "paquet faster-whisper absent de cet environnement"
    return True, None


def _build_faster_whisper(params: EngineParams):
    from ecoutemoi.core.engine_fw import FasterWhisperEngine

    return FasterWhisperEngine(params)


def _build_whispercpp(params: EngineParams, backend: str):
    from ecoutemoi.core.engine import WhisperEngine

    return WhisperEngine(dataclasses.replace(params, backend=backend))


def _cpu_engine(params: EngineParams, *, note: str | None = None):
    """Le moteur CPU : faster-whisper, ou whisper.cpp si le premier manque.

    `note` est la RAISON du basculement, pas un simple message de log : elle
    remonte jusqu'à l'opérateur (`engine.fallback_reason`). Découvrir en pleine
    conférence qu'on tourne sur un moteur qu'on n'a pas choisi est déjà
    désagréable ; l'apprendre sans savoir pourquoi est inexploitable.
    """
    ready, reason = _faster_whisper_ready(params)
    if ready:
        if note:
            log.info("%s — décodage CPU par faster-whisper.", note)
        engine = _build_faster_whisper(params)
    elif params.model_path is None:
        raise RuntimeError(f"Aucun moteur CPU disponible : {reason}, et aucun modèle ggml pour whisper.cpp.")
    else:
        log.warning("faster-whisper indisponible (%s) — repli sur whisper.cpp en CPU.", reason)
        note = (
            f"{note} ; faster-whisper indisponible ({reason})"
            if note
            else f"faster-whisper indisponible ({reason})"
        )
        engine = _build_whispercpp(params, "cpu")
    engine.fallback_reason = note
    return engine


def create_engine(params: EngineParams):
    """Construit le moteur adapté ; la surface publique est la même des deux côtés."""
    wanted = params.engine or "auto"

    if wanted == ENGINE_FASTER_WHISPER:
        ready, reason = _faster_whisper_ready(params)
        if not ready:
            log.warning("faster-whisper demandé mais indisponible (%s).", reason)
        return _cpu_engine(params)

    if wanted == ENGINE_WHISPERCPP:
        if params.model_path is None:
            return _cpu_engine(params, note="Modèle ggml absent")
        # L'échelle interne de whisper.cpp (GPU, GPU sans flash, CPU) reste le
        # bon comportement ici : l'opérateur a explicitement demandé ce moteur.
        return _build_whispercpp(params, params.backend)

    # ---------------------------------------------------------------- auto
    if params.backend == "cpu" or params.model_path is None:
        return _cpu_engine(params)

    try:
        # « gpu-only » : pas de barreau CPU. Un échec doit remonter ici pour que
        # le repli soit faster-whisper, pas le CPU (bien plus lent) de whisper.cpp.
        engine = _build_whispercpp(params, "gpu-only")
    except Exception as exc:
        return _cpu_engine(params, note=f"GPU indisponible ({exc})")
    if engine.gpu_active():
        return engine
    reason = engine.gpu_diagnostic() or "aucun périphérique GPU actif"
    engine.close()
    return _cpu_engine(params, note=f"GPU inactif ({reason})")


__all__ = ["create_engine", "engine_for_backend"]
