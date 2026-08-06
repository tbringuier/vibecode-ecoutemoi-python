"""diag_report : le rapport autonome derrière --diag et le dialogue GUI."""

import types

from ecoutemoi.cli import diag_report
from ecoutemoi.config import Settings


def _args(**kw):
    base = {"model": None, "backend": None, "gpu_device": None}
    return types.SimpleNamespace(**{**base, **kw})


def test_diag_environment_header_is_self_contained():
    lines, _ = diag_report(_args(model="nexistepas"), Settings())
    text = "\n".join(lines)
    # tout ce qu'un rapport de bug doit contenir sans question de relance
    for needle in ("diagnostic", "OS ", "Python", "Config", "Log", "Modèles", "Libs backend GPU"):
        assert needle in text


def test_diag_unknown_model_fails_clearly():
    lines, code = diag_report(_args(model="nexistepas"), Settings())
    assert code == 1
    assert any("Modèle inconnu" in line for line in lines)
