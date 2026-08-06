"""Le transcript : stockage en session, autosave continu, et écriture texte.

Deux usages, un seul format de données. Le direct alimente un `TranscriptStore`
qui écrit au fil de l'eau (une panne de courant ne doit pas coûter la
conférence) ; la transcription de fichiers, elle, produit ses segments d'un bloc
et n'a plus qu'à les rendre. Les deux passent par le même registre de formats,
donc un format ajouté est disponible des deux côtés.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import platformdirs

from ecoutemoi.config import atomic_write_text
from ecoutemoi.constants import APP_NAME

log = logging.getLogger(__name__)

AUTOSAVE_FLUSH_S = 10.0
MIN_SEGMENT_CHARS = 2  # segments shorter than this get merged into the previous one


@dataclass
class SessionSegment:
    """One finalized subtitle segment, times relative to session start."""

    t0_ms: int
    t1_ms: int
    text: str
    lang: str = "fr"


def default_sessions_root() -> Path:
    return platformdirs.user_documents_path() / APP_NAME / "sessions"


def new_session_dir(root: Path | None = None, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M")
    return (root or default_sessions_root()) / stamp


def _hms(ms: int) -> tuple[int, int, int, int]:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, mmm = divmod(rem, 1_000)
    return h, m, s, mmm


def fmt_srt_time(ms: int) -> str:
    h, m, s, mmm = _hms(ms)
    return f"{h:02d}:{m:02d}:{s:02d},{mmm:03d}"


def fmt_vtt_time(ms: int) -> str:
    h, m, s, mmm = _hms(ms)
    return f"{h:02d}:{m:02d}:{s:02d}.{mmm:03d}"


def fmt_txt_time(ms: int) -> str:
    h, m, s, _ = _hms(ms)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def to_txt(segments: list[SessionSegment], timestamps: bool = False) -> str:
    lines = []
    for seg in segments:
        prefix = f"{fmt_txt_time(seg.t0_ms)} " if timestamps else ""
        lines.append(f"{prefix}{seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def to_srt(segments: list[SessionSegment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(f"{i}\n{fmt_srt_time(seg.t0_ms)} --> {fmt_srt_time(seg.t1_ms)}\n{seg.text}\n")
    return "\n".join(blocks)


def to_vtt(segments: list[SessionSegment]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(f"{fmt_vtt_time(seg.t0_ms)} --> {fmt_vtt_time(seg.t1_ms)}\n{seg.text}\n")
    return "\n".join(blocks)


def to_markdown(segments: list[SessionSegment], timestamps: bool = True, title: str = "") -> str:
    """Transcript lisible et citable : un paragraphe par prise de parole.

    L'horodatage est mis en tête de paragraphe et en gras, pas en marge : c'est ce
    qui reste lisible dans un rendu Markdown quelconque, y compris collé dans un
    ticket ou un wiki.
    """
    lines: list[str] = []
    if title:
        lines += [f"# {title}", ""]
    for seg in segments:
        prefix = f"**{fmt_txt_time(seg.t0_ms)}** " if timestamps else ""
        lines += [f"{prefix}{seg.text}", ""]
    return "\n".join(lines).rstrip("\n") + "\n" if lines else ""


def to_json(segments: list[SessionSegment], title: str = "") -> str:
    """Format machine : segments intacts, millisecondes, sans perte."""
    from ecoutemoi import __version__

    payload = {
        "generator": f"EcouteMoi {__version__}",
        "source": title,
        "segment_count": len(segments),
        "segments": [asdict(seg) for seg in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _to_delimited(segments: list[SessionSegment], delimiter: str) -> str:
    """Tableur : début, fin, durée, langue, texte. Guillemets gérés par le module csv."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(["debut", "fin", "duree_s", "langue", "texte"])
    for seg in segments:
        writer.writerow(
            [
                fmt_srt_time(seg.t0_ms),
                fmt_srt_time(seg.t1_ms),
                f"{max(0, seg.t1_ms - seg.t0_ms) / 1000:.3f}",
                seg.lang,
                seg.text,
            ]
        )
    return buffer.getvalue()


def to_csv(segments: list[SessionSegment]) -> str:
    return _to_delimited(segments, ",")


def to_tsv(segments: list[SessionSegment]) -> str:
    return _to_delimited(segments, "\t")


def to_lrc(segments: list[SessionSegment], title: str = "") -> str:
    """Paroles synchronisées : lisible par la plupart des lecteurs audio.

    Utile pour relire une transcription EN ÉCOUTANT le fichier d'origine, ce que
    ni le TXT ni le SRT ne permettent hors d'un lecteur vidéo.
    """
    lines = [f"[ti:{title}]"] if title else []
    for seg in segments:
        centis = max(0, seg.t0_ms) // 10
        minutes, rest = divmod(centis, 6000)
        lines.append(f"[{minutes:02d}:{rest // 100:02d}.{rest % 100:02d}]{seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


@dataclass(frozen=True)
class TranscriptFormat:
    """Un format de sortie : son nom, ses extensions, son rendu."""

    key: str
    label: str
    extensions: tuple[str, ...]  # la première est celle qu'on écrit par défaut
    render: Callable[..., str] = field(compare=False, repr=False, default=lambda **_: "")
    timestamps: bool = False  # l'horodatage fait-il partie du format ?

    @property
    def extension(self) -> str:
        return self.extensions[0]

    @property
    def filter(self) -> str:
        """Entrée de filtre pour un sélecteur de fichiers Qt."""
        patterns = " ".join(f"*{ext}" for ext in self.extensions)
        return f"{self.label} ({patterns})"


# Ordre = ordre d'affichage dans l'interface, du plus courant au plus spécialisé.
TRANSCRIPT_FORMATS: dict[str, TranscriptFormat] = {
    fmt.key: fmt
    for fmt in (
        TranscriptFormat(
            "txt", "Texte brut", (".txt", ".text", ".log"),
            lambda segments, timestamps=False, title="": to_txt(segments, timestamps),
        ),
        TranscriptFormat(
            "md", "Markdown", (".md", ".markdown"),
            lambda segments, timestamps=True, title="": to_markdown(segments, timestamps, title),
            timestamps=True,
        ),
        TranscriptFormat(
            "srt", "Sous-titres SubRip", (".srt",),
            lambda segments, timestamps=True, title="": to_srt(segments),
            timestamps=True,
        ),
        TranscriptFormat(
            "vtt", "Sous-titres WebVTT", (".vtt", ".webvtt"),
            lambda segments, timestamps=True, title="": to_vtt(segments),
            timestamps=True,
        ),
        TranscriptFormat(
            "json", "JSON (segments horodatés)", (".json",),
            lambda segments, timestamps=True, title="": to_json(segments, title),
            timestamps=True,
        ),
        TranscriptFormat(
            "csv", "CSV (tableur)", (".csv",),
            lambda segments, timestamps=True, title="": to_csv(segments),
            timestamps=True,
        ),
        TranscriptFormat(
            "tsv", "TSV (tabulations)", (".tsv", ".tab"),
            lambda segments, timestamps=True, title="": to_tsv(segments),
            timestamps=True,
        ),
        TranscriptFormat(
            "lrc", "Paroles synchronisées LRC", (".lrc",),
            lambda segments, timestamps=True, title="": to_lrc(segments, title),
            timestamps=True,
        ),
    )
}  # fmt: skip

DEFAULT_FORMATS = ("txt", "srt")


def format_for_extension(extension: str) -> TranscriptFormat:
    """Format déduit d'une extension. Toute extension inconnue donne du TEXTE.

    C'est ce qui permet d'écrire dans « notes.dat » ou « conference.rtf » sans que
    l'application n'ait rien à dire : le contenu sera du texte brut, ce qui est la
    seule chose honnête à faire d'une extension qu'on ne connaît pas.
    """
    wanted = ("." + extension.lstrip(".")).lower()
    for fmt in TRANSCRIPT_FORMATS.values():
        if wanted in fmt.extensions:
            return fmt
    return TRANSCRIPT_FORMATS["txt"]


def format_filter() -> str:
    """Filtre complet pour un sélecteur d'enregistrement Qt."""
    return ";;".join([*(f.filter for f in TRANSCRIPT_FORMATS.values()), "Tous les fichiers (*)"])


def write_transcript(
    path: Path,
    segments: list[SessionSegment],
    *,
    key: str | None = None,
    timestamps: bool = False,
    title: str = "",
) -> Path:
    """Écrit le transcript. Le format vient de `key`, sinon de l'extension."""
    fmt = TRANSCRIPT_FORMATS.get(key or "") or format_for_extension(path.suffix)
    atomic_write_text(path, fmt.render(segments, timestamps=timestamps or fmt.timestamps, title=title))
    log.info("Transcript écrit : %s (%s, %d segments)", path, fmt.key, len(segments))
    return path


def output_path(source: Path, fmt: TranscriptFormat, directory: Path | None = None) -> Path:
    """Chemin de sortie pour un fichier source : même nom, autre extension."""
    return (directory or source.parent) / (source.stem + fmt.extension)


def unique_path(path: Path, taken: set[Path]) -> Path:
    """Évite d'écraser une sortie DÉJÀ écrite pendant ce même lot.

    « entretien.mp3 » et « entretien.m4a » visent tous deux « entretien.txt » :
    sans ce garde-fou, le second effacerait le premier en silence. Relancer le
    même fichier écrase par contre volontairement son propre résultat — c'est ce
    qu'on attend d'une seconde tentative.
    """
    if path not in taken:
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if candidate not in taken:
            return candidate
    return path


class TranscriptStore:
    """Holds finalized segments; appends to autosave files, flushes every 10 s.

    Autosave writes two files in the session directory:
    - ``transcript.txt``   human-readable, one line per segment
    - ``segments.jsonl``   machine-readable, allows SRT/VTT rebuild after a crash

    `suffix` sépare les fichiers de plusieurs flux d'une MÊME session (second
    sous-titre) : sans lui, les deux langues s'entremêleraient dans un seul
    transcript, ce qui le rendrait inutilisable pour les deux.
    """

    def __init__(self, session_dir: Path | None = None, autosave: bool = True, suffix: str = ""):
        self.session_dir = session_dir
        self.autosave = autosave and session_dir is not None
        self.suffix = suffix
        self.segments: list[SessionSegment] = []
        self._txt = None
        self._jsonl = None
        self._last_flush = time.monotonic()

    def _open_autosave(self) -> None:
        if not self.autosave or self._txt is not None:
            return
        assert self.session_dir is not None
        self.session_dir.mkdir(parents=True, exist_ok=True)
        # Long-lived append handles (continuous autosave), closed in close().
        self._txt = open(  # noqa: SIM115
            self.session_dir / f"transcript{self.suffix}.txt", "a", encoding="utf-8", newline="\n"
        )
        self._jsonl = open(  # noqa: SIM115
            self.session_dir / f"segments{self.suffix}.jsonl", "a", encoding="utf-8", newline="\n"
        )

    def add(self, segment: SessionSegment) -> None:
        text = segment.text.strip()
        if not text:
            return
        if len(text) < MIN_SEGMENT_CHARS and self.segments:
            prev = self.segments[-1]
            prev.text = f"{prev.text} {text}".strip()
            prev.t1_ms = max(prev.t1_ms, segment.t1_ms)
        else:
            segment.text = text
            self.segments.append(segment)
        if self.autosave:
            self._open_autosave()
            assert self._txt is not None and self._jsonl is not None
            self._txt.write(f"{fmt_txt_time(segment.t0_ms)} {text}\n")
            self._jsonl.write(json.dumps(asdict(segment), ensure_ascii=False) + "\n")
            now = time.monotonic()
            if now - self._last_flush >= AUTOSAVE_FLUSH_S:
                self.flush()

    def flush(self) -> None:
        for f in (self._txt, self._jsonl):
            if f is not None:
                f.flush()
        self._last_flush = time.monotonic()

    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    def export_txt(self, path: Path, timestamps: bool = False) -> None:
        atomic_write_text(path, to_txt(self.segments, timestamps))

    def export_srt(self, path: Path) -> None:
        atomic_write_text(path, to_srt(self.segments))

    def export_vtt(self, path: Path) -> None:
        atomic_write_text(path, to_vtt(self.segments))

    def export(self, path: Path, key: str | None = None, timestamps: bool = False) -> Path:
        """Export dans n'importe quel format du registre (ou d'après l'extension)."""
        return write_transcript(path, self.segments, key=key, timestamps=timestamps)

    def close(self) -> None:
        self.flush()
        for f in (self._txt, self._jsonl):
            if f is not None:
                f.close()
        self._txt = self._jsonl = None
