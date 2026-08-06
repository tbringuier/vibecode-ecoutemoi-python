"""Décodage de fichiers : formats, mono, rééchantillonnage, erreurs lisibles.

Les fichiers d'essai sont ÉCRITS par le test (libsndfile est une dépendance) :
aucun binaire n'entre dans le dépôt, et la couverture suit ce que la
bibliothèque livrée sait réellement faire sur la machine qui exécute les tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from ecoutemoi.constants import TARGET_SR
from ecoutemoi.core import media

sf = pytest.importorskip("soundfile")


def write_tone(path: Path, seconds: float = 1.0, sr: int = 44100, channels: int = 2) -> Path:
    """Sinusoïde à 220 Hz, canal droit atténué (pour vérifier le mixage mono)."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    left = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    data = left if channels == 1 else np.stack([left, left * 0.5], axis=1)
    sf.write(str(path), data.astype(np.float32), sr)
    return path


def test_probe_reads_what_matters(tmp_path):
    info = media.probe(write_tone(tmp_path / "a.wav", seconds=2.0))
    assert info.backend == "libsndfile"
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.duration_s == pytest.approx(2.0, abs=0.05)
    assert "44.1 kHz" in info.label and "2 canaux" in info.label


def test_decode_gives_float32_mono_16k(tmp_path):
    audio = media.decode_16k_mono(write_tone(tmp_path / "a.wav", seconds=1.5))
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(audio.size - int(1.5 * TARGET_SR)) < TARGET_SR // 10  # ±100 ms
    assert 0.05 < float(np.abs(audio).max()) <= 1.0


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".ogg", ".aiff"])
def test_common_formats_need_nothing_installed(tmp_path, suffix):
    """Le socle garanti : ces formats passent sans ffmpeg ni PyAV."""
    if suffix.lstrip(".") not in media._sndfile_formats():
        pytest.skip(f"libsndfile de cette machine ne gère pas {suffix}")
    info = media.probe(write_tone(tmp_path / f"a{suffix}"))
    assert info.backend == "libsndfile"
    assert media.decode_16k_mono(tmp_path / f"a{suffix}").size > TARGET_SR // 2


def test_streaming_decodes_in_bounded_blocks(tmp_path):
    """Le décodage rend des blocs : la mémoire ne suit pas la durée du fichier."""
    path = write_tone(tmp_path / "long.wav", seconds=25.0, sr=16000, channels=1)
    blocks = list(media.stream_16k_mono(path))
    assert len(blocks) > 1
    assert all(b.dtype == np.float32 for b in blocks)
    total = sum(b.size for b in blocks)
    assert abs(total - 25 * TARGET_SR) < TARGET_SR // 10


def test_cancellation_stops_decoding(tmp_path):
    path = write_tone(tmp_path / "long.wav", seconds=30.0, sr=16000, channels=1)
    seen: list[int] = []
    for block in media.stream_16k_mono(path, should_stop=lambda: len(seen) >= 1):
        seen.append(block.size)
    assert len(seen) == 1  # arrêt dès le bloc suivant


def test_missing_file_is_named(tmp_path):
    with pytest.raises(media.MediaError, match="introuvable"):
        media.probe(tmp_path / "absent.wav")


def test_unreadable_format_explains_how_to_fix(tmp_path, monkeypatch):
    """Sans décodeur capable, le message dit QUOI installer — pas « échec »."""
    monkeypatch.setattr(media, "ffmpeg_executable", lambda: None)
    monkeypatch.setattr(media, "pyav_available", lambda: False)
    fake = tmp_path / "vidéo.mkv"
    fake.write_bytes(b"pas un fichier audio")
    with pytest.raises(media.MediaError) as excinfo:
        media.probe(fake)
    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "vidéo.mkv" in message


def test_ffmpeg_path_override_is_taken_into_account(monkeypatch, tmp_path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n")
    try:
        media.set_ffmpeg_path(str(fake))
        assert media.ffmpeg_executable() == str(fake)
        media.set_ffmpeg_path("")  # retour à la détection automatique
        assert media.ffmpeg_executable() != str(fake)
    finally:
        media.set_ffmpeg_path("")


def test_dialog_filter_and_extensions():
    flt = media.file_dialog_filter()
    assert "*.mp3" in flt and "*.m4a" in flt and "*.mkv" in flt
    assert flt.count(";;") == 2  # audio ;; vidéo ;; tout
    assert media.looks_like_media(Path("a.MP3"))
    assert media.looks_like_media(Path("b.mkv"))
    assert not media.looks_like_media(Path("notes.txt"))


def test_duration_is_human_readable():
    assert media.fmt_duration(9) == "9 s"
    assert media.fmt_duration(75) == "1 min 15 s"
    assert media.fmt_duration(3 * 3600 + 5 * 60) == "3 h 05 min"


def test_decoder_report_lists_the_three_backends():
    report = media.decoder_report()
    assert len(report) == 3
    assert report[0].startswith("libsndfile")
    assert "ffmpeg" in report[1]
    assert "PyAV" in report[2]


# ------------------------------------------------------------------ voie ffmpeg
# ffmpeg n'est pas installé partout, et c'est pourtant LE décodeur des M4A et des
# fichiers vidéo — donc du cas Windows/macOS le plus courant. On le remplace par
# un script qui parle le même protocole : métadonnées sur stderr, PCM sur stdout.
FAKE_FFMPEG = """#!/bin/sh
for arg in "$@"; do
  if [ "$arg" = "f32le" ]; then cat "{raw}"; exit 0; fi
done
printf '  Duration: 00:00:02.50, start: 0.000000, bitrate: 128 kb/s\\n' >&2
printf '  Stream #0:0(und): Audio: aac (LC), 48000 Hz, stereo, fltp\\n' >&2
exit 1
"""


@pytest.fixture
def fake_ffmpeg(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("script shell : POSIX seulement")
    samples = np.linspace(-0.5, 0.5, 5 * TARGET_SR, dtype=np.float32)
    raw = tmp_path / "pcm.f32"
    raw.write_bytes(samples.tobytes())
    exe = tmp_path / "ffmpeg"
    exe.write_text(FAKE_FFMPEG.format(raw=raw))
    exe.chmod(0o755)
    monkeypatch.setattr(media, "pyav_available", lambda: False)  # forcer la voie ffmpeg
    media.set_ffmpeg_path(str(exe))
    yield samples
    media.set_ffmpeg_path("")


def test_ffmpeg_probe_reads_duration_and_codec(fake_ffmpeg, tmp_path):
    source = tmp_path / "entretien.m4a"
    source.write_bytes(b"m4a")
    info = media.probe(source)
    assert info.backend == "ffmpeg"
    assert info.codec == "aac"
    assert info.duration_s == pytest.approx(2.5)
    assert (info.sample_rate, info.channels) == (48000, 2)


def test_ffmpeg_decoding_reads_the_whole_pipe(fake_ffmpeg, tmp_path):
    source = tmp_path / "entretien.m4a"
    source.write_bytes(b"m4a")
    audio = media.decode_16k_mono(source)
    assert audio.dtype == np.float32
    assert audio.size == fake_ffmpeg.size  # rien perdu au découpage des blocs
    assert np.allclose(audio, fake_ffmpeg)


def test_ffmpeg_failure_surfaces_its_last_line(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("script shell : POSIX seulement")
    exe = tmp_path / "ffmpeg"
    exe.write_text("#!/bin/sh\necho 'Invalid data found when processing input' >&2\nexit 1\n")
    exe.chmod(0o755)
    monkeypatch.setattr(media, "pyav_available", lambda: False)
    media.set_ffmpeg_path(str(exe))
    try:
        info = media.MediaInfo(tmp_path / "x.m4a", "ffmpeg", None, None, None, "aac")
        with pytest.raises(media.MediaError, match="Invalid data"):
            media.decode_16k_mono(tmp_path / "x.m4a", info=info)
    finally:
        media.set_ffmpeg_path("")
