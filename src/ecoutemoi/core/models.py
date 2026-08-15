"""Model registry, downloads and manual import — DEUX formats.

Écoute Moi 2.0 a deux moteurs, donc deux formats de poids, incompatibles :

- **ggml** (`ggml-small-q5_1.bin`) pour whisper.cpp — c'est le format du GPU
  Vulkan/Metal. Un fichier par couple (famille, quantization).
- **CTranslate2** (dossier `ct2/small/`) pour faster-whisper — c'est le format
  du CPU. UN SEUL téléchargement par famille : la « quantization » y est un
  paramètre de chargement (`compute_type`), pas un fichier séparé. `small-q5_1`
  et `small-q8_0` partagent donc le même dossier `ct2/small/`.

Le registre reste indexé par les MÊMES clés qu'en 1.0 (`small-q5_1`…) : elles
vivent dans des settings.json et des bench_results.json déjà écrits.

Robustesse : chaque fichier est validé (magie ggml ou en-tête CTranslate2, plus
la taille à ±5 %) après téléchargement ET avant réutilisation — un fichier
tronqué ou corrompu est re-téléchargé automatiquement. Les téléchargements
réessaient avec backoff et l'espace disque est vérifié en amont (message clair
plutôt qu'un ENOSPC à 95 %).
"""

from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from ecoutemoi.constants import APP_NAME, MODEL_FORMATS

log = logging.getLogger(__name__)

WHISPER_REPO = "ggerganov/whisper.cpp"
VAD_REPO = "ggml-org/whisper-vad"
SIZE_TOLERANCE = 0.05  # ±5 %
DOWNLOAD_POOL = 2  # parallel downloads when several models are selected
DOWNLOAD_ATTEMPTS = 3  # tentatives réseau par modèle (backoff 2 s puis 5 s)
DOWNLOAD_BACKOFF_S = (2.0, 5.0)
GGML_MAGIC = b"lmgg"  # 0x67676d6c little-endian — tous les .bin whisper/silero
DISK_MARGIN_MB = 200  # marge au-delà de la taille du modèle

FMT_GGML = "ggml"  # whisper.cpp — GPU Vulkan/Metal (et CPU de secours)
FMT_CT2 = "ct2"  # faster-whisper — CPU

# Dépôts CTranslate2 officiels de faster-whisper (mêmes identifiants que sa
# table interne `faster_whisper.utils._MODELS` : pas de conversion maison, donc
# pas de divergence possible avec ce que le moteur sait charger).
CT2_REPOS: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Taille du dossier CTranslate2 par famille (MiB, poids f16 du dépôt amont).
# Ce sont les octets RÉELLEMENT téléchargés — pas la mémoire à l'exécution :
# CTranslate2 quantifie au chargement, donc `int8` occupe environ la moitié.
CT2_SIZE_MB: dict[str, int] = {
    "tiny": 75,
    "base": 141,
    "small": 464,
    "medium": 1460,
    "large-v2": 2947,
    "large-v3": 2948,
    "large-v3-turbo": 1547,
}

# Quantization ggml -> type de calcul CTranslate2. CTranslate2 ne descend pas
# sous 8 bits : q5_0 et q5_1 arrivent donc tous deux sur `int8`. La hiérarchie
# « plus compact -> plus fidèle » est préservée, ce qui compte pour l'opérateur
# qui compare deux lignes du benchmark.
QUANT_TO_COMPUTE: dict[str, str] = {
    "q5_0": "int8",
    "q5_1": "int8",
    "q8_0": "int8_float32",
    "f16": "float32",
}

# Fichiers qu'un dossier CTranslate2 doit contenir pour être chargeable.
CT2_REQUIRED = ("model.bin", "config.json", "tokenizer.json")
CT2_VOCAB = ("vocabulary.json", "vocabulary.txt")  # l'un OU l'autre selon le dépôt


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
    # --- format CTranslate2 (faster-whisper, CPU) ---
    ct2_repo: str = ""  # "" => pas de conversion CTranslate2 publiée
    ct2_size_mb: int = 0  # taille du dossier téléchargé (MiB)
    compute_type: str = "int8"  # type de calcul demandé à CTranslate2

    def size_mb_for(self, fmt: str) -> int:
        return self.ct2_size_mb if fmt == FMT_CT2 else self.size_mb

    def supports(self, fmt: str) -> bool:
        return bool(self.ct2_repo) if fmt == FMT_CT2 else True


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
    coquille ne se glisse dans un tableau de 17 lignes. Le versant CTranslate2
    (dépôt, taille, type de calcul) se déduit de la même façon, depuis la famille
    et la quantization.
    """
    stem = family if quant == "f16" else f"{family}-{quant}"
    return ModelSpec(stem, f"ggml-{stem}.bin", WHISPER_REPO, size_mb, translate,
                     ram_gb, role, family, quant,
                     ct2_repo=CT2_REPOS.get(family, ""),
                     ct2_size_mb=CT2_SIZE_MB.get(family, 0),
                     compute_type=QUANT_TO_COMPUTE.get(quant, "int8"))  # fmt: skip


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
    - >= 6 cœurs utiles et >= 8 Go de RAM : small-q5_1 ;
    - >= 4 cœurs utiles et >= 4 Go : base-q5_1 ;
    - sinon : tiny-q5_1.
    (« cœurs utiles » = P-cores physiques sur CPU hybride, sinon cœurs physiques.)

    Le seuil de `small` est descendu de 8 à 6 cœurs en 2.0 : faster-whisper
    décode `small` en ~870 ms sur une fenêtre de 9 s avec 6 P-cores (RTF ≈ 10),
    là où whisper.cpp mettait 2,9 s (RTF ≈ 3). Le pire cas — machine sans GPU —
    est donc largement au-dessus du plancher temps réel.
    """
    import psutil

    from ecoutemoi.core import cpuinfo

    cores = cpuinfo.best_n_threads()
    ram_gb = psutil.virtual_memory().total / 1e9
    if cores >= 6 and ram_gb >= 8:
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
    """Chemin du `.bin` ggml (whisper.cpp)."""
    return models_dir(base) / spec.filename


def ct2_dir(spec: ModelSpec, base: Path | None = None) -> Path:
    """Dossier CTranslate2 (faster-whisper) — UN par FAMILLE, pas par quantization.

    La quantization CTranslate2 se choisit au chargement (`compute_type`) : tous
    les `small-*` du registre partagent donc le même dossier `ct2/small/`, et
    l'opérateur ne télécharge pas trois fois le même modèle.
    """
    return models_dir(base) / "ct2" / (spec.family or spec.key)


def model_location(spec: ModelSpec, fmt: str, base: Path | None = None) -> Path:
    """Où vit ce modèle dans ce format : un fichier (ggml) ou un dossier (ct2)."""
    return ct2_dir(spec, base) if fmt == FMT_CT2 else model_path(spec, base)


def is_installed(spec: ModelSpec, base: Path | None = None, fmt: str | None = None) -> bool:
    """`fmt=None` => installé dans AU MOINS un des deux formats.

    C'est le sens utile pour lister des modèles dans l'interface : celui qui
    manque au backend courant sera récupéré au moment d'ouvrir la session.
    """
    if fmt is None:
        return any(is_installed(spec, base, f) for f in MODEL_FORMATS)
    if fmt == FMT_CT2:
        if not spec.ct2_repo:
            return False
        d = ct2_dir(spec, base)
        return d.is_dir() and (d / "model.bin").is_file() and (d / "model.bin").stat().st_size > 0
    p = model_path(spec, base)
    return p.is_file() and p.stat().st_size > 0


def installed_formats(spec: ModelSpec, base: Path | None = None) -> list[str]:
    return [f for f in MODEL_FORMATS if is_installed(spec, base, f)]


def installed_models(base: Path | None = None, fmt: str | None = None) -> list[str]:
    return [key for key, spec in REGISTRY.items() if is_installed(spec, base, fmt)]


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


def validate_ct2_dir(path: Path, spec: ModelSpec) -> str | None:
    """None si le dossier CTranslate2 est chargeable, sinon la raison.

    On vérifie la PRÉSENCE des fichiers indispensables et la taille des poids :
    un `snapshot_download` interrompu laisse volontiers un dossier à moitié
    peuplé, et CTranslate2 ne s'en plaindrait qu'à l'ouverture de la session.
    """
    if not path.is_dir():
        return "dossier absent"
    missing = [name for name in CT2_REQUIRED if not (path / name).is_file()]
    if missing:
        return f"fichier(s) manquant(s) : {', '.join(missing)}"
    if not any((path / name).is_file() for name in CT2_VOCAB):
        return f"vocabulaire absent ({' ou '.join(CT2_VOCAB)})"
    size = (path / "model.bin").stat().st_size
    if size == 0:
        return "model.bin vide"
    expected = spec.ct2_size_mb * 1024 * 1024
    if expected and abs(size - expected) > expected * (SIZE_TOLERANCE + 0.02):
        return f"model.bin hors tolérance ({size} octets pour ~{spec.ct2_size_mb} Mo)"
    return None


def validate_model(spec: ModelSpec, fmt: str, base: Path | None = None) -> str | None:
    """Validation d'un modèle dans le format demandé (None = sain)."""
    path = model_location(spec, fmt, base)
    return validate_ct2_dir(path, spec) if fmt == FMT_CT2 else validate_model_file(path, spec)


def _check_free_space(dest_dir: Path, spec: ModelSpec, fmt: str = FMT_GGML) -> None:
    """Échec précoce et lisible plutôt qu'un ENOSPC au milieu du téléchargement."""
    try:
        free_mb = shutil.disk_usage(dest_dir).free / (1024 * 1024)
    except OSError:
        return  # système de fichiers exotique : on laisse le téléchargement trancher
    needed_mb = spec.size_mb_for(fmt) + DISK_MARGIN_MB
    if free_mb < needed_mb:
        raise OSError(
            f"Espace disque insuffisant pour {spec.key} ({fmt}) : {free_mb:.0f} Mo libres, "
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


def download_ct2_model(spec: ModelSpec, base: Path | None = None, tqdm_class=None) -> Path:
    """Télécharge le dossier CTranslate2 d'une famille (faster-whisper).

    `snapshot_download` plutôt qu'un fichier à la fois : un modèle CTranslate2
    est un ENSEMBLE (poids + config + tokenizer + vocabulaire) dont aucun
    élément n'est facultatif. On écarte explicitement le README et les
    métadonnées git, qui ne servent à rien et gonflent le dossier.
    """
    if not spec.ct2_repo:
        raise ValueError(f"{spec.key} n'a pas de conversion CTranslate2 publiée")
    from huggingface_hub import snapshot_download

    dest = ct2_dir(spec, base)
    dest.mkdir(parents=True, exist_ok=True)
    _check_free_space(dest, spec, FMT_CT2)
    log.info("Téléchargement CTranslate2 %s (%s, ~%d Mo)…", spec.family, spec.ct2_repo, spec.ct2_size_mb)
    extra = {"tqdm_class": tqdm_class} if tqdm_class is not None else {}

    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(
                spec.ct2_repo,
                local_dir=str(dest),
                allow_patterns=["*.json", "*.txt", "model.bin"],
                ignore_patterns=["README.md", ".gitattributes"],
                **extra,
            )
            reason = validate_ct2_dir(dest, spec)
            if reason is None:
                log.info("Modèle CTranslate2 %s prêt : %s", spec.family, dest)
                return dest
            log.warning("Téléchargement CTranslate2 %s invalide (%s).", spec.family, reason)
            last_exc = OSError(f"dossier CTranslate2 invalide : {reason}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("CTranslate2 %s : tentative %d/%d échouée : %s",
                        spec.family, attempt, DOWNLOAD_ATTEMPTS, exc)  # fmt: skip
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(DOWNLOAD_BACKOFF_S[min(attempt - 1, len(DOWNLOAD_BACKOFF_S) - 1)])
    raise OSError(
        f"Téléchargement CTranslate2 de {spec.family} impossible après {DOWNLOAD_ATTEMPTS} "
        f"tentatives : {last_exc} — vérifiez la connexion (proxy ?) et l'espace disque, "
        f"ou copiez le dépôt {spec.ct2_repo} dans {dest}"
    ) from last_exc


def download(spec: ModelSpec, fmt: str, base: Path | None = None, tqdm_class=None) -> Path:
    """Téléchargement d'un modèle dans le format demandé."""
    if fmt == FMT_CT2:
        return download_ct2_model(spec, base, tqdm_class)
    return download_model(spec, base, tqdm_class)


def download_many(specs: list[ModelSpec], base: Path | None = None,
                  fmt: str = FMT_GGML) -> list[Path]:  # fmt: skip
    """Télécharge en parallèle ; les échecs n'annulent pas les autres modèles.

    En CTranslate2 les doublons de FAMILLE sont écartés : `small-q5_1` et
    `small-q8_0` désignent le même dossier, le télécharger deux fois en
    parallèle ne ferait que se marcher dessus.
    """
    results: dict[str, Path] = {}
    failures: dict[str, Exception] = {}
    todo: list[ModelSpec] = []
    seen: set[str] = set()
    for s in specs:
        if not s.supports(fmt):
            continue
        marker = (s.family or s.key) if fmt == FMT_CT2 else s.key
        if marker in seen:
            continue
        seen.add(marker)
        todo.append(s)

    def one(s: ModelSpec) -> None:
        try:
            results[s.key] = download(s, fmt, base)
        except Exception as exc:  # collecté, relancé en agrégat
            failures[s.key] = exc

    with ThreadPoolExecutor(max_workers=DOWNLOAD_POOL) as pool:
        list(pool.map(one, todo))
    if failures:
        detail = " ; ".join(f"{k}: {e}" for k, e in failures.items())
        raise OSError(f"{len(failures)} téléchargement(s) en échec — {detail}")
    return [results[s.key] for s in todo if s.key in results]


def ensure_model(key: str, base: Path | None = None, fmt: str = FMT_GGML) -> Path:
    """Chemin du modèle dans ce format, en le (re)téléchargeant si absent OU corrompu."""
    spec = REGISTRY[key]
    if fmt == FMT_CT2 and not spec.ct2_repo:
        raise ValueError(f"{key} n'a pas de conversion CTranslate2 publiée")
    path = model_location(spec, fmt, base)
    reason = validate_model(spec, fmt, base)
    if reason is None:
        return path
    if fmt == FMT_GGML and path.is_file():
        log.warning("Modèle %s invalide sur disque (%s) — re-téléchargement.", key, reason)
        path.unlink(missing_ok=True)
    elif fmt == FMT_CT2 and path.exists():
        log.warning("Modèle CTranslate2 %s incomplet (%s) — complété.", spec.family, reason)
    return download(spec, fmt, base)


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
