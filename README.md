# Écoute Moi

> **Ce logiciel est intégralement « vibecodé »** — écrit en dialoguant avec un
> modèle de langage, du premier commit au dernier. Il est **fourni en l'état, sans
> aucune garantie** (fonctionnement, exactitude des transcriptions, adéquation à un
> usage), les binaires ne sont pas signés, et il **est et doit rester gratuit**.
> Une transcription automatique se trompe : en contexte d'accessibilité elle ne
> remplace ni un vélotypiste ni un interprète. L'avertissement complet s'affiche au
> premier lancement et reste consultable dans **Aide → Avertissement et licence**.

Deux usages, **100 % en local**, sans compte et sans réseau :

**Sous-titrer en direct.** Un orateur, un micro, des sous-titres dans une fenêtre
à fond vert que capture OBS (incrustation chromatique), avec prévisualisation dans
la fenêtre principale. Modes **FR→FR**, **FR→EN** et **Auto→EN** (langue parlée
détectée en continu, sous-titres toujours traduits en anglais).

**Transcrire des enregistrements.** Des fichiers audio ou vidéo, quel qu'en soit
le codec, vers du texte : brut, Markdown, SRT, WebVTT, JSON, CSV, TSV, LRC — ou
n'importe quelle autre extension.

**Deux moteurs, chacun là où il gagne** (depuis la 2.0) : **faster-whisper**
(CTranslate2) sur **CPU**, **whisper.cpp** sur **GPU** Vulkan ou Metal. Le
backend se choisit d'un menu, ou se laisse en « Auto ». Rien n'est envoyé nulle
part, sauf si vous activez explicitement une sortie réseau.

## Télécharger et lancer (binaires non signés — assumé)

Les trois binaires sont publiés **en fichiers bruts** sur la page
[Releases](../../releases/latest) :

| OS | Fichier | Premier lancement |
|---|---|---|
| Windows x64 | `EcouteMoi.exe` — **un seul fichier**, Python embarqué, auto-extractible | SmartScreen : « Informations complémentaires » → **Exécuter quand même** |
| Linux x86_64 | `EcouteMoi-linux-x86_64.AppImage` | `chmod +x` puis lancer ; sans FUSE : `./EcouteMoi-*.AppImage --appimage-extract-and-run` |
| macOS arm64 (M-series) | `EcouteMoi-macos-arm64.zip` → `EcouteMoi.app` | clic droit → **Ouvrir**, ou `xattr -cr EcouteMoi.app` ; autoriser le **micro** |

Au premier démarrage, deux étapes qui ne se font qu'une fois :

1. **Outils → Gérer les modèles…** — téléchargement rapide, reprise incluse.
2. **Outils → Benchmark — Tester ma machine…** — vous lisez deux courts textes
   (FR/EN), l'application mesure vitesse (RTF) et qualité (WER) de chaque modèle
   installé, abandonne les trop lents et **applique la recommandation en un clic**.
   Cinq minutes ici évitent de découvrir en salle que le modèle choisi ne tient pas
   le direct.

**Aide → Prise en main** (F1) résume la mise en route en une page.

## Transcrire des fichiers audio

**Fichier → Transcrire des fichiers audio…** (Ctrl+O), ou déposez les fichiers sur
la fenêtre, ou passez-les en argument (`EcouteMoi conference.mp3`) — ce qui fait
marcher « Ouvrir avec… » depuis le gestionnaire de fichiers.

Hors direct, rien ne presse : **prenez le modèle le plus lourd que la machine
accepte**, la qualité s'en ressentira. Une transcription deux fois plus lente
qu'un direct reste bien plus rapide que le temps réel.

### Formats d'entrée

| Décodeur | Ce qu'il couvre | À installer |
|---|---|---|
| libsndfile (livrée) | WAV, FLAC, **MP3**, OGG/Vorbis, Ogg Opus, AIFF, CAF, W64, AU | rien |
| ffmpeg, s'il est présent | M4A/AAC, WMA, AMR, et les pistes audio de MP4, MKV, MOV, WebM, TS | `winget install --id Gyan.FFmpeg`, `brew install ffmpeg`, ou le paquet de votre distribution |
| **PyAV (livrée depuis la 2.0)** | idem ffmpeg, en bibliothèque | rien — faster-whisper en dépend, elle est donc embarquée |

Depuis la 2.0, installer ffmpeg n'est plus nécessaire : PyAV arrive avec
faster-whisper et couvre les mêmes conteneurs. Un ffmpeg système reste préféré
quand il est là, parce qu'il est souvent plus récent.

Sous Windows, déposer `ffmpeg.exe` **à côté de `EcouteMoi.exe`** suffit : il est
trouvé là aussi. Un chemin explicite peut être indiqué dans
**Réglages avancés → Fichiers**, qui affiche également ce que la machine sait lire.

### Formats de sortie

`txt` · `md` · `srt` · `vtt` · `json` · `csv` · `tsv` · `lrc` — cochez ceux que
vous voulez, ils sont écrits d'un coup. Toute **autre** extension est acceptée et
reçoit du texte brut : `notes.dat` fonctionne, et c'est la seule chose honnête à
faire d'une extension inconnue. `ecoutemoi --list-formats` en donne la liste.

Le découpage des passes de décodage tombe **là où personne ne parle** : couper au
milieu d'un mot le fait perdre des deux côtés, whisper devinant différemment de
part et d'autre de la coupe.

En ligne de commande :

```bash
ecoutemoi --transcribe conference.mp3 entretien.m4a --to srt,md --out ./textes
```

Un seul chargement de modèle pour tout le lot ; un fichier illisible est signalé
et n'interrompt pas les autres ; le code de retour ne vaut 0 que si tout est passé.

## Modèles et quantizations

18 variantes multilingues, de `tiny-q5_1` (31 Mo en ggml) à `large-v2-q8_0`
(1,6 Go) — toutes celles qui peuvent tenir le temps réel sur une machine ou une
autre, pour pouvoir les comparer au benchmark. Les poids ggml viennent du dépôt
officiel `ggerganov/whisper.cpp`, les conversions CTranslate2 des dépôts
`Systran/faster-whisper-*` (et `mobiuslabsgmbh` pour `large-v3-turbo`), ceux-là
mêmes qu'utilise faster-whisper — pas de conversion maison.

**Aide → Quantizations — aide-mémoire** explique ce que valent `q5_0`, `q5_1`,
`q8_0` et `f16`, comment choisir selon CPU ou GPU, et à quel type de calcul
CTranslate2 chacun correspond (CTranslate2 ne descend pas sous 8 bits : `q5_0`
et `q5_1` y arrivent tous deux sur `int8`).

Deux familles sont volontairement absentes : les `large` en `f16` (2,9 Go, hors
budget temps réel sur des fenêtres de 9 s) et les modèles `.en` (anglais
uniquement et sans traduction — aucun des trois modes ne pourrait les employer).

Un modèle peut aussi être installé **sans téléchargement** : déposez le `.bin`
ggml dans le dossier des modèles, ou **Gérer les modèles… → Importer un .bin
ggml…** (le nom du fichier identifie le modèle, le contenu est validé avant
installation). En CLI : `--import-model FICHIER`. Un modèle CTranslate2 est un
*dossier* : copiez-le dans `models/ct2/<famille>/`.

En CLI, `--download` récupère par défaut le format dont le backend réglé a
besoin ; `--format {ggml,ct2,both}` le force.

## Lexique de la conférence

**Direct → Lexique de la conférence…** (Ctrl+L) : les noms propres, produits et
acronymes du talk, un par ligne. Ils sont passés à whisper comme amorce de
transcription à *chaque* fenêtre décodée (`initial_prompt` +
`carry_initial_prompt`). C'est le levier le plus efficace sur exactement ce que la
quantization dégrade en premier — les mots rares. En CLI :
`--lexicon "Kubernetes, Ceph, OpenStack"`.

## Le texte ne se répète pas

Un sous-titre qui redit la même phrase est pire qu'un sous-titre absent : le
public croit avoir manqué quelque chose et cherche la différence entre les deux
occurrences. Le flux temps réel a trois sources de répétition, traitées
séparément (`core/textguard.py`) :

- **le recouvrement.** La coupe de fenêtre garde 200 ms d'audio déjà décodé, et la
  pré-amorce du détecteur de parole rejoue 300 ms à chaque reprise. Les mots déjà
  publiés sont mémorisés et retirés du début de l'hypothèse suivante — la mémoire
  survit à la fin d'un énoncé, puisque c'est justement là que le doublon arrive.
- **le bégaiement du décodeur.** « d'accord d'accord d'accord » : le motif répété
  est détecté quelle que soit sa longueur, avec une exception pour les doublés
  légitimes du français (« nous nous »).
- **le segment servi deux fois**, même texte et autre horodatage : le second est
  écarté.

En amont, la **traîne de silence** qui a déclenché la fin d'un énoncé n'est pas
donnée au modèle : une fenêtre qui finit par deux secondes de vide, whisper la
remplit — en répétant la phrase précédente, ou en inventant un « merci d'avoir
regardé ». Une fenêtre entièrement muette n'est pas décodée du tout.

**Les mots non encore validés ne sont jamais affichés.** LocalAgreement-2 retient
le mot en cours de prononciation jusqu'à confirmation : le public ne voit que du
texte définitif, jamais un mot qui se corrige sous ses yeux. Il n'existe aucun
canal dans le code pour transporter du texte non validé hors du streamer, et deux
tests verrouillent l'invariant.

## Double sous-titre

Cocher **Second sous-titre** dans « Modèle et langue » ouvre un deuxième flux
complet, avec son mode et son modèle : deuxième fenêtre de sortie
(`EcouteMoi - Sortie OBS 2`, titre distinct pour la capture OBS), deuxième
transcript (`transcript_2.txt`), deuxième source OBS, deuxième cadencement. Un
seul détecteur de parole alimente les deux moteurs, donc la capture et le DSP ne
sont pas payés deux fois — mais **deux moteurs tournent** : la RAM se cumule et le
voyant temps réel prend le pire des deux canaux.

## Deux moteurs, un seul réglage

Aucun moteur ne couvre bien tout le terrain, alors l'application en embarque deux
et prend le meilleur de chacun :

| | Moteur | Pourquoi lui |
|---|---|---|
| **GPU** Vulkan (Intel/AMD/NVIDIA) ou Metal (Apple) | **whisper.cpp** | c'est le seul des deux à savoir parler Vulkan et Metal. CTranslate2, sous faster-whisper, ne connaît que le CPU : sur un GPU, quel qu'en soit le vendeur, il n'existe pas. |
| **CPU** | **faster-whisper** (CTranslate2) | environ **trois fois plus rapide** que whisper.cpp sur CPU, à qualité égale ou meilleure. |

Mesuré sur un Intel Core Ultra 9 185H (6 P-cores) + Arc iGPU, fenêtre de 9 s,
médiane de six décodages, modèle `small` :

| Moteur / backend | Décodage | RTF | WER (voix de référence FR) |
|---|---|---|---|
| whisper.cpp / CPU | 3 529 ms | 2,6 | 8,5 % |
| **faster-whisper / CPU** | **931 ms** | **9,7** | 8,5 % |
| whisper.cpp / Vulkan | 754 ms | 11,9 | 8,5 % |

Le cas le plus parlant est `medium`, où le changement n'est pas quantitatif mais
qualitatif :

| Moteur / backend (`medium-q5_0`) | Décodage | RTF | Verdict |
|---|---|---|---|
| whisper.cpp / CPU | 9 894 ms | **0,91** | 🔴 décroche — plus lent que le direct |
| **faster-whisper / CPU** | **2 866 ms** | **3,14** | 🟢 fluide |
| whisper.cpp / Vulkan | 2 093 ms | 4,30 | 🟢 fluide |

Autrement dit : sur une machine sans GPU utilisable, `small` passe de « tout
juste tenable » à « confortable », et `medium` d'**impossible** à fluide. Le GPU,
lui, garde sa longueur d'avance — celle qu'on aurait perdue en remplaçant
whisper.cpp partout, puisque CTranslate2 n'a pas de backend Vulkan ni Metal.

### Le réglage

Dans la fenêtre principale (« Backend ») ou dans **Réglages avancés** :

- **Auto** — GPU si un périphérique répond, sinon CPU ;
- **GPU** — force whisper.cpp sur Vulkan (Windows/Linux) ou Metal (macOS), avec
  avertissement et repli CPU si aucun GPU n'est utilisable ;
- **CPU** — faster-whisper, sans jamais toucher au GPU.

En CLI : `--backend {auto,gpu,cpu}`. Le moteur et le backend réellement actifs
sont affichés dans la barre d'état, dans `--diag` et dans les logs ; en cas de
repli, une notice dit **lequel** tourne et **pourquoi** l'autre a été écarté.

**Réglages avancés** permet aussi de forcer whisper.cpp sur CPU (repli si
faster-whisper pose problème sur une machine) et de choisir la précision de
calcul CTranslate2 (`int8`, `int8_float32`, `float32`).

### Deux moteurs, deux formats de modèle

whisper.cpp lit du **ggml** (`ggml-small-q5_1.bin`), faster-whisper du
**CTranslate2** (un dossier). Aucun des deux ne sait lire le format de l'autre.
Conséquences visibles :

- **Gérer les modèles…** a un sélecteur de format, et une colonne « Installé »
  qui montre en permanence les deux ;
- en « Auto », l'application **sonde le GPU au premier lancement** (quelques
  dizaines de millisecondes, sans ouvrir de modèle, résultat mémorisé) pour ne
  télécharger *que* le format dont elle a besoin. `--reprobe-gpu` refait le test
  après un changement de pilote ;
- un dossier CTranslate2 est **partagé par toutes les quantizations d'une même
  famille** : la quantization y est un paramètre de chargement, pas un fichier.
  Télécharger `small-q5_1` installe donc aussi `small-q8_0` et `small`.

### Et ma carte NVIDIA / AMD / Intel ?

**Vulkan les couvre toutes les trois**, sans rien installer : son loader est
livré avec le pilote graphique. C'est pour ça qu'il est le chemin GPU principal
et non un pis-aller — un seul backend, tous les vendeurs.

CUDA, ROCm, OpenVINO et oneAPI ne sont pas proposés : chacun exige une
installation de plusieurs gigaoctets sur la machine de l'utilisateur, ou
n'accélère que l'encodeur à partir d'un modèle converti hors ligne. Pour un
matériel que Vulkan sert déjà immédiatement.

### Contexte d'encodeur adapté à la fenêtre

L'encodeur de whisper traite **toujours 30 s** de spectrogramme, même quand on
ne lui donne que 9 s : les 21 s de vide sont calculées plein tarif, plusieurs
fois par seconde. Depuis la 2.0 le contexte est tronqué à ce dont la fenêtre a
besoin (whisper.cpp uniquement — CTranslate2 n'expose pas ce réglage).

Mesuré de bout en bout sur le pipeline complet, `small-q5_1`, 6 énoncés,
23 décodages, reproductible au millième sur trois passes :

| échantillon · backend | contexte plein | adapté | gain | WER |
|---|---|---|---|---|
| FR · Vulkan | 520 ms | **278 ms** | ×1,9 | 6,74 % → 6,38 % |
| FR · CPU whisper.cpp | 2855 ms | **894 ms** | ×3,2 | 6,74 % → 6,03 % |
| EN · Vulkan | 480 ms | **225 ms** | ×2,1 | 0,00 % → 0,00 % |
| EN · CPU whisper.cpp | 2711 ms | **826 ms** | ×3,3 | 0,00 % → 0,00 % |

WER égal ou meilleur dans les quatre cas. La **transcription de fichiers est
inchangée** : ses passes de 25 s utilisent déjà le contexte entier, la sortie
est identique octet pour octet.

Les deux voix de référence sont de la synthèse vocale. Si vous constatez des
mots manquants en fin de phrase sur une vraie captation, la case
*Réglages avancés → Contexte d'encodeur adapté à la fenêtre* le désactive.

### Ce qui a été mesuré et écarté

- **Décodage par lots** (`BatchedInferencePipeline` de faster-whisper), sur 258 s
  d'audio : `small` **aucun gain** (10,0 s → 10,1 s), `medium` +18 % de vitesse
  mais **WER de 8,5 % à 10,5 %**. Le lot sert à remplir un GPU ; sur CPU,
  CTranslate2 sature déjà les cœurs avec une seule séquence. Non retenu.
- **Faisceau de décodage sur fichier** (beam 5), même mesure : +35 % de temps
  pour un **WER de 8,5 % à 5,9 %**. Retenu — hors direct, personne n'attend.
- **Plus de threads sur CPU hybride** : 6 P-cores donnent 839 ms sur une fenêtre
  de 9 s, 10 fils 988 ms, **16 fils 1728 ms** — deux fois pire. Les E-cores
  freinent le calcul au lieu de l'aider. La politique « P-cores physiques,
  jamais le SMT » est confirmée telle quelle.
- **Raccourcir l'encodeur de faster-whisper** : sans effet (1478 / 1383 /
  1416 ms pour 30 / 15 / 10 s, soit du bruit). CTranslate2 rembourre en
  interne — le chemin CPU est à son maximum.

### Côté machine

Vulkan ne demande que les pilotes GPU : rien à installer sous Windows (le loader
`vulkan-1.dll` accompagne le pilote) ; sous Linux, les pilotes Vulkan de la
distribution (ex. `mesa-vulkan-drivers`) — vérifiables avec `vulkaninfo --summary`.

Sur CPU, le moteur utilise par défaut autant de threads que de **P-cores
physiques** (CPU hybrides Intel/Apple), sinon tous les cœurs physiques — jamais
l'hyperthreading. Au premier lancement, le modèle par défaut est **profilé sur
la machine** (cœurs utiles + RAM : tiny / base / small) ; le benchmark guidé
affine ensuite, en mesurant le moteur qui tournera vraiment.

## Sorties : OBS et page web

Trois voies, cumulables. Les trois affichent **exactement les mêmes lignes** que
la fenêtre de sortie.

**1. Capture de fenêtre + chroma key** (par défaut, aucune configuration). Menu
**Aide → Guide OBS (par système)**. En bref : Capture de fenêtre sur
« `EcouteMoi - Sortie OBS` » + filtre Incrustation chromatique (similarité ≈ 400,
lissage ≈ 80, réduction du débordement ≈ 100). Fond magenta disponible si votre
contenu contient du vert.

**2. Texte poussé par WebSocket** (**Réglages avancés → Diffusion**). Dans OBS :
*Outils → Paramètres du serveur WebSocket* → activer le serveur (port 4455), puis
ajouter une source **Texte (GDI+)**. Renseignez hôte/port/mot de passe, cliquez
**Tester la connexion** (la liste des sources texte se remplit), choisissez la
source. Plus de chroma key : OBS compose le texte lui-même, donc net à toute
taille, sans liseré, et stylable dans OBS. La reconnexion est automatique — OBS
peut démarrer après l'application, ou redémarrer pendant la session.

**3. Page web locale** (**Réglages avancés → Diffusion**), servie sur
`http://127.0.0.1:8777/`. À ouvrir dans un navigateur (second écran, retour
orateur, régie) ou comme **Source navigateur** OBS. Ajoutez
`?bg=transparent` à l'URL pour un fond transparent au lieu du fond chroma.
Option « accessible depuis le réseau local » pour un poste de régie distant —
sans authentification, donc à réserver aux réseaux de confiance. Raccourci :
**Outils → Ouvrir la page web des sous-titres**.

## Lisibilité : débit constant et largeur de ligne

Whisper valide les mots par **rafales** : sans lissage, deux lignes surgissent d'un
bloc puis rien pendant deux secondes — le texte est juste, mais le public n'a pas
le temps de le lire. **Réglages avancés → Sous-titrage** applique donc les normes
du métier (BBC Subtitle Guidelines, EBU-TT-D) :

- **débit de lecture borné** à 180 mots/min par défaut, avec un plafond de retard
  au-delà duquel le débit accélère pour revenir à niveau — sans ce garde-fou, un
  orateur rapide creuserait un décalage sans fin ;
- **largeur de ligne bornée** à 42 caractères, au-delà l'œil perd la ligne au
  retour chariot même sur une fenêtre large.

Le retard introduit volontairement est affiché en bas de la fenêtre (« Cadence »).

## Voyant temps réel

En bas à gauche, un voyant clignotant dit si la machine tient le direct, d'après
le RTF médian des derniers décodages (le pire des deux canaux en double
sous-titre) :

| Voyant | Signification | Quoi faire |
|---|---|---|
| 🟢 vert | RTF ≥ 3 — marge confortable | rien |
| 🟠 ambre | RTF ≥ 1,3 — ça tient, sans marge | surveiller ; éviter les autres charges |
| 🔴 rouge | RTF < 1,3 — le direct décroche | modèle plus petit, preset Phrase, ou GPU |

Le moteur est **chargé et préchauffé dès l'ouverture de l'application** (en tâche de
fond, sans jamais télécharger un modèle absent) : le surcoût unique du premier
décodage — compilation des shaders Vulkan, allocation du graphe — est payé pendant
que vous réglez votre police, donc « Démarrer » est immédiat. À défaut, il est payé
avant l'ouverture du micro, pendant « Préchauffage du moteur… », jamais pendant la
session : sinon l'audio s'accumule et sort en rafale de rattrapage.

Le décodage tourne par défaut dans un **sous-processus dédié** : plus de gigue GIL
sur le rendu ni sur le callback audio, et un crash natif du moteur (pilote Vulkan,
modèle corrompu) est survivable — le processus est relancé et le modèle rechargé.

## Développement

Prérequis : [uv](https://docs.astral.sh/uv/) (Python 3.14 installé par uv).

```bash
uv sync
uv run pytest -m "not integration"
uv run ecoutemoi            # interface graphique
```

> **faster-whisper** (le moteur CPU) s'installe tel quel depuis PyPI : rien à
> compiler. La wheel **pywhispercpp** de PyPI, en revanche, est compilée **sans**
> backend GPU — en dev, le backend « gpu » retombe donc sur le CPU. Pour le GPU
> en dev, compilez la wheel locale : `bash scripts/build_wheel.sh` (Linux,
> paquets `cmake glslc libvulkan-dev`) ou `scripts/build_wheel.ps1` (Windows,
> SDK Vulkan LunarG), puis
> `uv pip install wheelhouse/pywhispercpp-*.whl --force-reinstall`.
> Ensuite **toujours** `uv run --no-sync` : un `uv sync` restaurerait la wheel CPU.

### CLI (débogage, mesures, serveurs)

```bash
uv run ecoutemoi --list-devices
uv run ecoutemoi --list-models                                   # registre, les 2 formats
uv run ecoutemoi --download small-q5_1                           # format du backend réglé
uv run ecoutemoi --download small-q5_1 --format both             # ggml ET CTranslate2
uv run ecoutemoi --cli --model small-q5_1 --mode fr --preset stable
uv run ecoutemoi --transcribe discours.mp3 --to srt,txt          # hors direct
uv run ecoutemoi --cli --wav discours.wav --rate realtime        # simulation direct
uv run ecoutemoi --cli --backend cpu                             # faster-whisper
uv run ecoutemoi --cli --backend gpu                             # whisper.cpp Vulkan/Metal
uv run ecoutemoi --rtf tiny-q5_1,base-q5_1 --wav calibration_fr.wav
uv run ecoutemoi --bench --wav-fr fr.wav --wav-en en.wav         # bench complet + JSON
uv run ecoutemoi --diag                                          # moteurs, GPU, modèles
uv run ecoutemoi --check-engines                                 # les 2 moteurs sont-ils là ?
uv run ecoutemoi --reprobe-gpu                                   # re-tester le GPU
```

Seul le texte **validé** est imprimé en mode console. Réglages persistés (JSON
atomique) ; sessions (autosave + exports) dans `Documents/EcouteMoi/sessions/`. La
console n'affiche que les avertissements — le détail va dans le fichier de log,
`--verbose` le remet à l'écran.

> **Nettoyage micro : « Aucun traitement » par défaut.** Mesuré au WER (clip JFK,
> modèle base, bruit blanc ajouté) : whisper transcrit *mieux* l'audio brut que
> l'audio débruité, même à 5 dB de SNR (0-4,5 % de WER brut contre ~32 %
> débruité). Whisper est entraîné sur de l'audio bruité ; les artefacts du
> débruitage le perturbent davantage que le bruit. Le passe-haut 80 Hz et
> RNNoise restent disponibles pour les cas extrêmes (grondement de scène,
> souffle constant type ventilation).

### Wheels moteur GPU

Les binaires publiés embarquent un moteur pywhispercpp compilé depuis les sources
avec `GGML_VULKAN=1` et le jeu d'instructions du variant avx2 canonique de ggml
(`AVX2+FMA+F16C+BMI2`, portable Haswell 2013+ / Zen 2017+) sur Windows/Linux,
et Metal sur macOS arm64. Si Vulkan n'est pas utilisable au lancement (pas de
GPU, pilote cassé), le moteur retombe automatiquement sur le CPU.
En local : `scripts/build_wheel.ps1` (Vulkan SDK LunarG requis) ou `scripts/build_wheel.sh`.

L'AppImage n'embarque volontairement ni libstdc++ ni libgcc : les pilotes Vulkan
de l'hôte se chargent dans le processus et exigent la version de l'hôte (une
libstdc++ embarquée plus vieille ferait échouer leur chargement, silencieusement,
et l'application resterait en CPU).

### Tests

- `uv run pytest -m "not integration"` — unitaires (rapides, sans réseau ; les
  tests des sorties OBS/web montent un faux serveur obs-websocket en local) ;
- `uv run pytest -m integration` — téléchargement tiny + inférence réelle,
  endurance 10 min, dégradation automatique, interface graphique hors écran.

### Packaging

`packaging/ecoutemoi.spec` (PyInstaller) : **onefile** sur Windows (un seul
`.exe`), onedir + AppImage sur Linux, onedir + `.app` sur macOS. La bibliothèque
libsndfile y est embarquée explicitement — `soundfile` est un module et non un
paquet, les collecteurs automatiques de PyInstaller ne la voient pas.

Le second moteur alourdit le binaire : `libctranslate2`, `tokenizers`,
`onnxruntime` (le VAD Silero interne de faster-whisper) et PyAV s'ajoutent aux
libs ggml. Deux garde-fous font échouer le build plutôt que de livrer un bundle
amputé — l'un exige `ggml-vulkan` (sinon l'artefact n'aurait jamais de GPU),
l'autre `libctranslate2` et le `.onnx` Silero (sinon le moteur CPU serait absent
ou muet). Le premier est contournable par `ECOUTEMOI_ALLOW_CPU_BUNDLE=1`, en
connaissance de cause.

## Licence

**GPL-3.0-or-later** — voir [LICENSE](LICENSE). Copyleft assumé : toute
redistribution, modifiée ou non, doit rester libre et sous la même licence. C'est
la traduction juridique de « ce logiciel est et doit rester gratuit ».

Les modèles Whisper sont © OpenAI et suivent leurs propres licences ; les
conversions sont publiées par ggml-org (ggml) et Systran / mobiuslabs
(CTranslate2). L'interface utilise Qt via PySide6 (LGPL, bibliothèques
dynamiques) ; libsndfile est sous LGPL-2.1-or-later ; faster-whisper et
CTranslate2 sont sous MIT ; PyAV sous BSD-3-Clause (et embarque ffmpeg, LGPL).

Journal des versions : [CHANGELOG.md](CHANGELOG.md).
