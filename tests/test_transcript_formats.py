"""Formats de sortie exacts, UTF-8, règle de fusion, autosave, registre."""

import json
from pathlib import Path

from ecoutemoi.core.transcript import (
    SessionSegment,
    TranscriptStore,
    fmt_srt_time,
    fmt_txt_time,
    fmt_vtt_time,
    to_srt,
    to_txt,
    to_vtt,
)

SEGS = [
    SessionSegment(0, 2500, "Bonjour à tous."),
    SessionSegment(3_661_234, 3_662_000, "Éléphant à l'école."),
]


def test_time_formats():
    assert fmt_srt_time(0) == "00:00:00,000"
    assert fmt_srt_time(3_661_234) == "01:01:01,234"
    assert fmt_vtt_time(3_661_234) == "01:01:01.234"
    assert fmt_txt_time(3_661_234) == "[01:01:01]"
    assert fmt_srt_time(-5) == "00:00:00,000"  # clamped


def test_srt_exact():
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Bonjour à tous.\n"
        "\n"
        "2\n"
        "01:01:01,234 --> 01:01:02,000\n"
        "Éléphant à l'école.\n"
    )
    assert to_srt(SEGS) == expected


def test_vtt_exact():
    expected = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\n"
        "Bonjour à tous.\n"
        "\n"
        "01:01:01.234 --> 01:01:02.000\n"
        "Éléphant à l'école.\n"
    )
    assert to_vtt(SEGS) == expected


def test_txt_with_and_without_timestamps():
    assert to_txt(SEGS) == "Bonjour à tous.\nÉléphant à l'école.\n"
    assert to_txt(SEGS, timestamps=True) == ("[00:00:00] Bonjour à tous.\n[01:01:01] Éléphant à l'école.\n")
    assert to_txt([]) == ""


def test_short_segment_merged():
    store = TranscriptStore(None, autosave=False)
    store.add(SessionSegment(0, 1000, "Bonjour tout le monde"))
    store.add(SessionSegment(1000, 1200, "à"))
    assert len(store.segments) == 1
    assert store.segments[0].text == "Bonjour tout le monde à"
    assert store.segments[0].t1_ms == 1200


def test_empty_segment_dropped():
    store = TranscriptStore(None, autosave=False)
    store.add(SessionSegment(0, 100, "   "))
    assert store.segments == []


def test_exports_utf8_atomic(tmp_path):
    store = TranscriptStore(None, autosave=False)
    for seg in SEGS:
        store.add(SessionSegment(seg.t0_ms, seg.t1_ms, seg.text))
    srt = tmp_path / "out.srt"
    vtt = tmp_path / "out.vtt"
    txt = tmp_path / "out.txt"
    store.export_srt(srt)
    store.export_vtt(vtt)
    store.export_txt(txt, timestamps=True)
    # UTF-8 accents intact
    assert "Éléphant à l'école." in srt.read_text(encoding="utf-8")
    assert vtt.read_bytes().startswith(b"WEBVTT")
    assert "[01:01:01]" in txt.read_text(encoding="utf-8")
    # no leftover tmp files from the atomic write
    assert list(tmp_path.glob("*.tmp")) == []


def test_autosave_files(tmp_path):
    session = tmp_path / "session"
    store = TranscriptStore(session, autosave=True)
    store.add(SessionSegment(0, 1500, "Première phrase complète."))
    store.add(SessionSegment(1600, 3000, "Deuxième phrase complète."))
    store.close()
    txt = (session / "transcript.txt").read_text(encoding="utf-8")
    assert "[00:00:00] Première phrase complète." in txt
    lines = (session / "segments.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["text"] == "Deuxième phrase complète."


def test_word_count():
    store = TranscriptStore(None, autosave=False)
    store.add(SessionSegment(0, 1000, "un deux trois"))
    assert store.word_count() == 3


# ------------------------------------------------- registre de formats de sortie
def test_registry_covers_the_announced_formats():
    from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS

    assert set(TRANSCRIPT_FORMATS) == {"txt", "md", "srt", "vtt", "json", "csv", "tsv", "lrc"}
    for fmt in TRANSCRIPT_FORMATS.values():
        assert fmt.extension.startswith(".")
        assert fmt.filter.startswith(fmt.label)
        assert fmt.render(SEGS, timestamps=True, title="source.mp3")


def test_markdown_and_lrc():
    from ecoutemoi.core.transcript import to_lrc, to_markdown

    md = to_markdown(SEGS, timestamps=True, title="conférence.mp3")
    assert md.startswith("# conférence.mp3\n")
    assert "**[00:00:00]** Bonjour à tous." in md
    assert "**[01:01:01]** Éléphant à l'école." in md
    assert to_markdown(SEGS, timestamps=False).startswith("Bonjour à tous.")
    assert to_markdown([]) == ""

    lrc = to_lrc(SEGS, title="conférence.mp3")
    assert lrc.splitlines()[0] == "[ti:conférence.mp3]"
    assert lrc.splitlines()[1] == "[00:00.00]Bonjour à tous."
    assert lrc.splitlines()[2] == "[61:01.23]Éléphant à l'école."


def test_json_keeps_milliseconds():
    from ecoutemoi.core.transcript import to_json

    data = json.loads(to_json(SEGS, title="a.wav"))
    assert data["source"] == "a.wav"
    assert data["segment_count"] == 2
    assert data["segments"][1] == {
        "t0_ms": 3_661_234,
        "t1_ms": 3_662_000,
        "text": "Éléphant à l'école.",
        "lang": "fr",
    }
    assert data["generator"].startswith("EcouteMoi ")


def test_csv_and_tsv_quote_properly():
    from ecoutemoi.core.transcript import SessionSegment as Seg
    from ecoutemoi.core.transcript import to_csv, to_tsv

    tricky = [Seg(0, 1000, 'Il a dit : "bonjour", puis rien.')]
    csv_text = to_csv(tricky)
    assert csv_text.splitlines()[0] == "debut,fin,duree_s,langue,texte"
    assert '"Il a dit : ""bonjour"", puis rien."' in csv_text
    tsv_text = to_tsv(SEGS)
    assert tsv_text.splitlines()[1].split("\t")[4] == "Bonjour à tous."


def test_unknown_extension_falls_back_to_plain_text(tmp_path):
    from ecoutemoi.core.transcript import format_for_extension, write_transcript

    assert format_for_extension(".dat").key == "txt"
    assert format_for_extension("MARKDOWN").key == "md"
    assert format_for_extension(".webvtt").key == "vtt"
    target = write_transcript(tmp_path / "notes.dat", SEGS)
    assert target.read_text(encoding="utf-8") == "Bonjour à tous.\nÉléphant à l'école.\n"


def test_write_transcript_honours_the_extension(tmp_path):
    from ecoutemoi.core.transcript import write_transcript

    srt = write_transcript(tmp_path / "out.srt", SEGS)
    assert srt.read_text(encoding="utf-8").startswith("1\n00:00:00,000")
    # `key` gagne sur l'extension quand l'appelant l'impose
    forced = write_transcript(tmp_path / "out.txt", SEGS, key="vtt")
    assert forced.read_text(encoding="utf-8").startswith("WEBVTT")


def test_output_and_unique_paths(tmp_path):
    from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS, output_path, unique_path

    srt = TRANSCRIPT_FORMATS["srt"]
    assert output_path(Path("/a/b/entretien.mp3"), srt) == Path("/a/b/entretien.srt")
    assert output_path(Path("/a/b/entretien.mp3"), srt, tmp_path) == tmp_path / "entretien.srt"
    # Deux sources de même nom dans un lot ne s'écrasent pas
    taken = {tmp_path / "entretien.srt"}
    second = unique_path(tmp_path / "entretien.srt", taken)
    assert second == tmp_path / "entretien (2).srt"
    assert unique_path(tmp_path / "autre.srt", taken) == tmp_path / "autre.srt"


def test_store_export_uses_the_registry(tmp_path):
    store = TranscriptStore(None, autosave=False)
    for seg in SEGS:
        store.add(SessionSegment(seg.t0_ms, seg.t1_ms, seg.text))
    path = store.export(tmp_path / "session.md", timestamps=True)
    assert "**[00:00:00]**" in path.read_text(encoding="utf-8")
