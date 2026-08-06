"""Détection de la topologie CPU : P-cores (CPU hybrides) et cœurs physiques.

Politique de threads du moteur : autant de threads que de P-cores physiques
(jamais les E-cores, jamais le SMT/hyperthreading) ; à défaut de topologie
hybride détectable, tous les cœurs physiques.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_cpu_list(text: str) -> list[int]:
    """Parse une liste sysfs du type "0-11" ou "0-3,8-11"."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _p_cores_linux() -> int | None:
    """Intel hybride : /sys/devices/cpu_core liste les CPU logiques des P-cores."""
    cpus_file = Path("/sys/devices/cpu_core/cpus")
    if not cpus_file.is_file():
        return None
    try:
        logical = _parse_cpu_list(cpus_file.read_text())
        cores: set[tuple[str, str]] = set()
        for c in logical:
            topo = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
            pkg = (topo / "physical_package_id").read_text().strip()
            core = (topo / "core_id").read_text().strip()
            cores.add((pkg, core))
        return len(cores) or None
    except OSError, ValueError:
        return None


def _p_cores_macos() -> int | None:
    """Apple Silicon : hw.perflevel0 = cœurs performance (absent sur Mac Intel)."""

    def sysctl(name: str) -> int:
        res = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True)
        return int(res.stdout.strip())

    try:
        if sysctl("hw.nperflevels") < 2:
            return None
        return sysctl("hw.perflevel0.physicalcpu") or None
    except Exception:
        return None


def _p_cores_windows() -> int | None:
    """GetLogicalProcessorInformationEx : un enregistrement RelationProcessorCore
    par cœur physique, EfficiencyClass la plus haute = P-cores."""
    import ctypes
    from ctypes import wintypes

    RELATION_PROCESSOR_CORE = 0
    try:
        k32 = ctypes.windll.kernel32
        needed = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None, ctypes.byref(needed))
        buf = (ctypes.c_ubyte * needed.value)()
        if not k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf, ctypes.byref(needed)):
            return None
        # Enregistrements de taille variable : DWORD Relationship, DWORD Size,
        # puis PROCESSOR_RELATIONSHIP { BYTE Flags; BYTE EfficiencyClass; ... }.
        classes: list[int] = []
        offset = 0
        raw = bytes(buf)
        while offset + 8 <= needed.value:
            relationship = int.from_bytes(raw[offset : offset + 4], "little")
            size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
            if size <= 0:
                return None
            if relationship == RELATION_PROCESSOR_CORE:
                classes.append(raw[offset + 9])
            offset += size
        if not classes:
            return None
        top = max(classes)
        if top == 0:  # CPU homogène : pas de notion de P/E
            return None
        return sum(1 for c in classes if c == top)
    except Exception:
        return None


def p_core_count() -> int | None:
    """Nombre de P-cores physiques, ou None si le CPU n'est pas hybride."""
    if sys.platform.startswith("linux"):
        return _p_cores_linux()
    if sys.platform == "darwin":
        return _p_cores_macos()
    if sys.platform == "win32":
        return _p_cores_windows()
    return None


def physical_cores() -> int:
    """Cœurs physiques (SMT exclu) ; replis successifs jusqu'à 4."""
    import psutil

    return psutil.cpu_count(logical=False) or psutil.cpu_count() or 4


def best_n_threads() -> int:
    """P-cores physiques si CPU hybride, sinon tous les cœurs physiques."""
    p = p_core_count()
    if p:
        log.debug("CPU hybride : %d P-cores physiques", p)
        return p
    return physical_cores()
