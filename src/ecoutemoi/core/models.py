"""Model registry, downloads and manual import.

Robustesse : chaque fichier est validé (magie ggml + taille ±5 %) après
téléchargement ET avant réutilisation — un fichier tronqué ou corrompu est
re-téléchargé automatiquement. Les téléchargements réessaient avec backoff et
l'espace disque est vérifié en amont (message clair plutôt qu'un ENOSPC à 95 %).
"""

from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from ecoutemoi.constants import APP_NAME

log = logging.getLogger(__name__)

WHISPER_REPO = "ggerganov/whisper.cpp"
VAD_REPO = "ggml-org/whisper-vad"
SIZE_TOLERANCE = 0.05  # ±5 %
DOWNLOAD_POOL = 2  # parallel downloads when several models are selected
DOWNLOAD_ATTEMPTS = 3  # tentatives réseau par modèle (backoff 2 s puis 5 s)
DOWNLOAD_BACKOFF_S = (2.0, 5.0)
GGML_MAGIC = b"lmgg"  # 0x67676d6c little-endian — tous les .bin whisper/silero
DISK_MARGIN_MB = 200  # marge au-delà de la taille du modèle


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    repo_id: str
    size_mb: int  # approximate size in MiB (±5 % checked after download)
    translate: bool  # False => task=translate unsupported (large-v3-turbo)
    ram_gb: float  # model + activations estimate, shown in the UI
    role: str  # short FR description for the UI
    family: str = ""  # tiny | base | small | medium | large-v3-turbo | large-v2 | large-v3
    quant: str = "f16"  # f16 | q5_0 | q5_1 | q8_0 — voir QUANT_NOTES


# Quantizations disponibles pour un modèle donné, de la plus compacte à la plus
# fidèle. Repères mesurés sur whisper.cpp ; le détail lisible par l'opérateur
# vit dans QUANT_NOTES (aide-mémoire de la GUI).
QUANT_ORDER = ("q5_0", "q5_1", "q8_0", "f16")

QUANT_NOTES: dict[str, tuple[str, str]] = {
    "q5_0": (
        "≈ 35 % de la taille f16",
        "5 bits par poids, un facteur d'échelle par bloc. Le plus compact ici. "
        "Perte réelle mais discrète : elle se voit surtout sur les noms propres "
        "et les mots rares. Seule quantization proposée pour medium et large.",
    ),
    "q5_1": (
        "≈ 40 % de la taille f16",
        "5 bits, avec un décalage EN PLUS du facteur d'échelle par bloc : "
        "légèrement plus gros que q5_0, sensiblement plus fidèle. À préférer "
        "quand les deux existent (tiny, base, small).",
    ),
    "q8_0": (
        "≈ 54 % de la taille f16",
        "8 bits : quasiment sans perte, indiscernable de f16 à l'oreille sur du "
        "français courant. Le bon choix dès que la RAM et la bande passante "
        "suivent — c'est-à-dire sur GPU.",
    ),
    "f16": (
        "taille de référence",
        "Poids d'origine en demi-précision, aucune perte de quantization. "
        "Référence qualité, mais 2 à 3 fois plus de mémoire à lire par "
        "décodage : sur CPU c'est souvent le facteur limitant.",
    ),
}


def _spec(family: str, quant: str, size_mb: int, translate: bool, ram_gb: float, role: str) -> ModelSpec:
    """Un modèle du dépôt ggml officiel ; `quant='f16'` = précision d'origine.

    La clé et le nom de fichier sont DÉRIVÉS (pas recopiés) : c'est ce qui garantit
    que `key` reste stable pour les settings/benchs déjà enregistrés et qu'aucune
    coquille ne se glisse dans un tableau de 17 lignes.
    """
    stem = family if quant == "f16" else f"{family}-{quant}"
    return ModelSpec(stem, f"ggml-{stem}.bin", WHISPER_REPO, size_mb, translate,
                     ram_gb, role, family, quant)  # fmt: skip


# Toutes les variantes MULTILINGUES du dépôt ggml plausibles en temps réel, pour
# pouvoir comparer les quantizations au benchmark. Tailles = tailles RÉELLES du
# dépôt Hugging Face en MiB (la validation post-téléchargement tolère ±5 %).
#
# Volontairement ABSENTS :
# - large-v1 / large-v2 / large-v3 en f16 (2,9 Go chacun) : aucune machine ne les
#   tient en temps réel sur des fenêtres de 9 s, ils ne feraient que gonfler la
#   liste et le benchmark ;
# - les variantes `.en` (anglais uniquement) : elles ne savent ni transcrire le
#   français ni traduire, donc aucun des trois modes (FR→FR, FR→EN, Auto→EN) ne
#   peut les utiliser. À ajouter en même temps qu'un mode EN→EN, pas avant ;
# - q4_* : la dégradation devient audible sur les noms propres, sans gain de
#   vitesse notable face à q5_0.
REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        # tiny — latence plancher, qualité limitée
        _spec("tiny", "q5_1", 31, True, 0.4, "latence minimale absolue"),
        _spec("tiny", "q8_0", 42, True, 0.4, "tiny quasi sans perte"),
        _spec("tiny", "f16", 74, True, 0.5, "tiny en référence (comparaison de quantization)"),
        # base — le bon compromis sur petits CPU
        _spec("base", "q5_1", 57, True, 0.5, "meilleur ratio latence/qualité petits CPU"),
        _spec("base", "q8_0", 78, True, 0.5, "base quasi sans perte"),
        _spec("base", "f16", 141, True, 0.6, "base en référence (comparaison de quantization)"),
        # small — référence conférence
        _spec("small", "q5_1", 181, True, 0.9, "défaut si aucun bench"),
        _spec("small", "q8_0", 252, True, 1.0, "small quasi sans perte (GPU/CPU à l'aise)"),
        _spec("small", "f16", 465, True, 1.2, "qualité FR un cran au-dessus"),
        # medium — haute qualité, exige de la marge
        _spec("medium", "q5_0", 514, True, 1.6, "haute qualité si RTF ≥ 3 au bench"),
        _spec("medium", "q8_0", 785, True, 1.9, "medium quasi sans perte (GPU conseillé)"),
        _spec("medium", "f16", 1463, True, 2.5, "medium en référence (GPU obligatoire)"),
        # large-v3-turbo — décodeur distillé : vitesse de medium, qualité de large,
        # MAIS entraîné sans la tâche de traduction (il ressortirait la langue source).
        _spec("large-v3-turbo", "q5_0", 547, False, 1.7, "FR→FR haut de gamme (pas de traduction)"),
        _spec("large-v3-turbo", "q8_0", 834, False, 2.0,
              "turbo quasi sans perte, FR→FR (pas de traduction)"),
        _spec("large-v3-turbo", "f16", 1549, False, 2.7,
              "turbo en référence, FR→FR (pas de traduction, GPU obligatoire)"),
        # large complets quantifiés — qualité maximale AVEC traduction, GPU requis
        _spec("large-v2", "q5_0", 1031, True, 2.3, "qualité maximale traduisible (GPU obligatoire)"),
        _spec("large-v2", "q8_0", 1579, True, 2.8, "large-v2 quasi sans perte (GPU rapide)"),
        _spec("large-v3", "q5_0", 1031, True, 2.3, "large-v3 traduisible (GPU obligatoire)"),
    )
}  # fmt: skip

VAD_SPEC = ModelSpec("silero-vad", "ggml-silero-v6.2.0.bin", VAD_REPO, 1, False, 0.0,
                     "VAD Silero interne whisper.cpp (obligatoire)", "silero", "f16")  # fmt: skip


def recommended_default_model() -> str:
    """Modèle par défaut profilé sur la machine (premier lancement, avant tout bench).

    Heuristique CPU volontairement prudente — le benchmark guidé affine ensuite :
    - >= 8 cœurs utiles et >= 8 Go de RAM : small-q5_1 ;
    - >= 4 cœurs utiles et >= 4 Go : base-q5_1 ;
    - sinon : tiny-q5_1.
    (« cœurs utiles » = P-cores physiques sur CPU hybride, sinon cœurs physiques.)
    """
    import psutil

    from ecoutemoi.core import cpuinfo

    cores = cpuinfo.best_n_threads()
    ram_gb = psutil.virtual_memory().total / 1e9
    if cores >= 8 and ram_gb >= 8:
        choice = "small-q5_1"
    elif cores >= 4 and ram_gb >= 4:
        choice = "base-q5_1"
    else:
        choice = "tiny-q5_1"
    log.info("Profil machine : %d cœurs utiles, %.1f Go RAM -> modèle par défaut %s",
             cores, ram_gb, choice)  # fmt: skip
    return choice


def models_dir(base: Path | None = None) -> Path:
    return (base or platformdirs.user_data_path(APP_NAME, appauthor=False)) / "models"


def model_path(spec: ModelSpec, base: Path | None = None) -> Path:
    return models_dir(base) / spec.filename


def is_installed(spec: ModelSpec, base: Path | None = None) -> bool:
    p = model_path(spec, base)
    return p.is_file() and p.stat().st_size > 0


def installed_models(base: Path | None = None) -> list[str]:
    return [key for key, spec in REGISTRY.items() if is_installed(spec, base)]


def check_size(path: Path, spec: ModelSpec) -> bool:
    """±5 % size sanity check; specs of ~1 MiB (VAD) only get a floor."""
    size = path.stat().st_size
    expected = spec.size_mb * 1024 * 1024
    if spec.size_mb <= 2:
        return size > 300_000
    return abs(size - expected) <= expected * SIZE_TOLERANCE


def check_magic(path: Path) -> bool:
    """Les .bin whisper ET silero commencent par la magie ggml (vérifié sur pièce)."""
    try:
        with path.open("rb") as f:
            return f.read(4) == GGML_MAGIC
    except OSError:
        return False


def validate_model_file(path: Path, spec: ModelSpec) -> str | None:
    """None si le fichier est sain, sinon la raison (pour log + re-téléchargement)."""
    if not path.is_file():
        return "fichier absent"
    if path.stat().st_size == 0:
        return "fichier vide"
    if not check_magic(path):
        return "magie ggml absente (fichier corrompu ou tronqué)"
    if not check_size(path, spec):
        return f"taille hors tolérance ±5 % ({path.stat().st_size} octets pour ~{spec.size_mb} Mo)"
    return None


def _check_free_space(dest_dir: Path, spec: ModelSpec) -> None:
    """Échec précoce et lisible plutôt qu'un ENOSPC au milieu du téléchargement."""
    try:
        free_mb = shutil.disk_usage(dest_dir).free / (1024 * 1024)
    except OSError:
        return  # système de fichiers exotique : on laisse le téléchargement trancher
    needed_mb = spec.size_mb + DISK_MARGIN_MB
    if free_mb < needed_mb:
        raise OSError(
            f"Espace disque insuffisant pour {spec.key} : {free_mb:.0f} Mo libres, "
            f"~{needed_mb} Mo nécessaires ({dest_dir})"
        )


def import_model_file(source: Path, base: Path | None = None) -> tuple[str, Path]:
    """Installe un .bin fourni à la main. Retourne (clé du modèle, chemin installé).

    Le nom de fichier fait foi : c'est lui qui identifie le modèle dans le
    registre. Un fichier renommé serait indétectable — mieux vaut refuser avec la
    liste des noms attendus que d'installer quelque chose d'inconnu.
    """
    name = source.name
    matches = [spec for spec in (*REGISTRY.values(), VAD_SPEC) if spec.filename == name]
    if not matches:
        raise ValueError(
            f"« {name} » ne correspond à aucun modèle du registre. Noms attendus : "
            + ", ".join(sorted(spec.filename for spec in REGISTRY.values()))
            + f", {VAD_SPEC.filename} (VAD)."
        )
    spec = matches[0]
    reason = validate_model_file(source, spec)
    if reason is not None:
        raise ValueError(f"« {name} » invalide : {reason}")
    dest_dir = models_dir(base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / name
    if source.resolve() == destination.resolve():
        return spec.key, destination  # déjà en place
    shutil.copy2(source, destination)
    log.info("Modèle importé : %s -> %s", source, destination)
    return spec.key, destination


def _resolve_vad_filename() -> str:
    """The VAD filename may move forward (silero vX.Y.Z); fall back to the newest one."""
    from huggingface_hub import HfApi

    files = [
        f for f in HfApi().list_repo_files(VAD_REPO) if f.startswith("ggml-silero") and f.endswith(".bin")
    ]
    if not files:
        raise FileNotFoundError(f"No ggml-silero-*.bin found in {VAD_REPO}")
    files.sort()
    log.warning("VAD %s not found; falling back to newest available: %s", VAD_SPEC.filename, files[-1])
    return files[-1]


def _download_once(spec: ModelSpec, dest_dir: Path, tqdm_class) -> Path:
    """Un téléchargement hf_hub (reprise/xet gérés par huggingface_hub)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    extra = {"tqdm_class": tqdm_class} if tqdm_class is not None else {}
    try:
        return Path(hf_hub_download(spec.repo_id, spec.filename, local_dir=dest_dir, **extra))
    except EntryNotFoundError:
        if spec.key != VAD_SPEC.key:
            raise
        return Path(hf_hub_download(spec.repo_id, _resolve_vad_filename(), local_dir=dest_dir, **extra))


def download_model(spec: ModelSpec, base: Path | None = None, tqdm_class=None) -> Path:
    """Download one model, with retries + integrity validation.

    - espace disque vérifié en amont (message clair) ;
    - 3 tentatives avec backoff (réseau instable, proxy capricieux) ;
    - fichier validé (magie ggml + taille) et SUPPRIMÉ s'il est corrompu, pour
      que la tentative suivante reparte proprement.
    `tqdm_class`: custom progress class -> Qt signal in the model manager.
    """
    dest_dir = models_dir(base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _check_free_space(dest_dir, spec)
    log.info("Downloading %s (%s, ~%d Mo)...", spec.key, spec.filename, spec.size_mb)

    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            path = _download_once(spec, dest_dir, tqdm_class)
            reason = validate_model_file(path, spec)
            if reason is None:
                log.info("Model %s ready: %s", spec.key, path)
                return path
            log.warning("Téléchargement %s invalide (%s) — fichier supprimé.", spec.key, reason)
            path.unlink(missing_ok=True)
            last_exc = OSError(f"fichier téléchargé invalide : {reason}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("Téléchargement %s : tentative %d/%d échouée : %s",
                        spec.key, attempt, DOWNLOAD_ATTEMPTS, exc)  # fmt: skip
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(DOWNLOAD_BACKOFF_S[min(attempt - 1, len(DOWNLOAD_BACKOFF_S) - 1)])
    raise OSError(
        f"Téléchargement de {spec.key} impossible après {DOWNLOAD_ATTEMPTS} tentatives : {last_exc} "
        f"— vérifiez la connexion (proxy ?) et l'espace disque, ou importez le fichier "
        f"{spec.filename} manuellement dans {dest_dir}"
    ) from last_exc


def download_many(specs: list[ModelSpec], base: Path | None = None) -> list[Path]:
    """Télécharge en parallèle ; les échecs n'annulent pas les autres modèles."""
    results: dict[str, Path] = {}
    failures: dict[str, Exception] = {}

    def one(s: ModelSpec) -> None:
        try:
            results[s.key] = download_model(s, base)
        except Exception as exc:  # collecté, relancé en agrégat
            failures[s.key] = exc

    with ThreadPoolExecutor(max_workers=DOWNLOAD_POOL) as pool:
        list(pool.map(one, specs))
    if failures:
        detail = " ; ".join(f"{k}: {e}" for k, e in failures.items())
        raise OSError(f"{len(failures)} téléchargement(s) en échec — {detail}")
    return [results[s.key] for s in specs]


def ensure_model(key: str, base: Path | None = None) -> Path:
    """Chemin du modèle, en le (re)téléchargeant si absent OU corrompu."""
    spec = REGISTRY[key]
    p = model_path(spec, base)
    reason = validate_model_file(p, spec) if p.is_file() else "fichier absent"
    if reason is None:
        return p
    if p.is_file():
        log.warning("Modèle %s invalide sur disque (%s) — re-téléchargement.", key, reason)
        p.unlink(missing_ok=True)
    return download_model(spec, base)


def _any_silero_on_disk(base: Path | None) -> Path | None:
    """N'importe quel silero valide déjà présent (repli, ou import manuel)."""
    candidates = [c for c in sorted(models_dir(base).glob("ggml-silero-*.bin")) if check_magic(c)]
    return candidates[-1] if candidates else None


def ensure_vad_model(base: Path | None = None) -> Path:
    p = model_path(VAD_SPEC, base)
    if p.is_file() and validate_model_file(p, VAD_SPEC) is None:
        return p
    try:
        return download_model(VAD_SPEC, base)
    except Exception:
        fallback = _any_silero_on_disk(base)
        if fallback is not None:
            return fallback
        raise
