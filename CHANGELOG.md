# Journal des versions

## 2.0.0 — 15 août 2026

**Changement de moteur.** L'application en embarque désormais **deux**, et prend
le meilleur de chacun plutôt que d'en imposer un partout :

- **faster-whisper** (CTranslate2) sur **CPU** — nouveau ;
- **whisper.cpp** sur **GPU** Vulkan et Metal — conservé, parce que CTranslate2
  ne connaît que le CPU : sur un GPU, quel qu'en soit le vendeur, il n'existe
  pas.

Mesuré sur Intel Core Ultra 9 185H (6 P-cores) + Arc iGPU, fenêtre de 9 s,
médiane de six décodages :

| modèle | whisper.cpp CPU | faster-whisper CPU | whisper.cpp Vulkan |
|---|---|---|---|
| `tiny-q5_1` | 472 ms (RTF 19,1) | **213 ms** (RTF 42,3) | 260 ms (RTF 34,6) |
| `base-q5_1` | 941 ms (RTF 9,6) | **426 ms** (RTF 21,1) | 370 ms (RTF 24,3) |
| `small-q5_1` | 3 529 ms (RTF 2,6) | **931 ms** (RTF 9,7) | 754 ms (RTF 11,9) |
| `medium-q5_0` | 9 894 ms (RTF **0,91**) | **2 866 ms** (RTF 3,14) | 2 093 ms (RTF 4,3) |

Le gain n'est pas seulement quantitatif : sur une machine sans GPU utilisable,
`small` passe de « tout juste tenable » à « confortable », et `medium` de
**plus lent que le direct** à fluide. Le WER de faster-whisper est égal ou
meilleur à chaque modèle mesuré.

- **Backend** : « CPU » désigne maintenant faster-whisper, « GPU » whisper.cpp,
  « Auto » choisit. La barre d'état, `--diag` et les notices disent **lequel**
  tourne et **pourquoi** l'autre a été écarté.
- **Sondage GPU** au premier lancement (`core/gpuprobe`) : quelques dizaines de
  millisecondes, sans ouvrir de modèle, dans un sous-processus, résultat
  mémorisé. Il sert à ne télécharger *que* le format de modèle nécessaire.
  `--reprobe-gpu` le refait après un changement de pilote.
- **Deux formats de modèle** : ggml (whisper.cpp) et CTranslate2
  (faster-whisper). « Gérer les modèles… » gagne un sélecteur de format et une
  colonne « Installé » montrant les deux ; `--list-models` aussi ;
  `--download --format {ggml,ct2,both}`. Un dossier CTranslate2 est partagé par
  toutes les quantizations d'une même famille.
- **Réglages avancés** : moteur CPU (faster-whisper ou repli whisper.cpp) et
  précision de calcul CTranslate2 (`int8`, `int8_float32`, `float32`).
- **Transcription de fichiers** : décodage en faisceau (beam 5) hors direct, où
  aucune latence n'est en jeu ; le direct reste glouton.
- **PyAV embarquée** : faster-whisper en dépend, elle n'est donc plus optionnelle
  — installer ffmpeg n'est plus nécessaire pour les conteneurs vidéo.
- Le modèle par défaut profilé sur la machine passe à `small` dès 6 cœurs utiles
  (contre 8), le CPU n'étant plus le facteur limitant qu'il était.

**Un seul chemin GPU, assumé : Vulkan** (Metal sur Apple). Il couvre NVIDIA, AMD
et Intel sans que l'utilisateur installe quoi que ce soit — son loader vient avec
le pilote. CUDA, ROCm, OpenVINO et oneAPI ne sont pas proposés : chacun exige une
installation de plusieurs gigaoctets, ou n'accélère que l'encodeur depuis un
modèle converti hors ligne, pour du matériel que Vulkan sert déjà.

- `--check-engines` vérifie après empaquetage que les deux moteurs s'importent :
  une exclusion PyInstaller qui casse une chaîne d'imports ne produit aucune
  erreur — juste un décodage trois fois plus lent, en silence. (C'est arrivé
  pendant ce développement.)

**Mesuré puis écarté.** Le décodage par lots de faster-whisper
(`BatchedInferencePipeline`) sur 258 s d'audio : aucun gain sur `small`, +18 %
sur `medium` mais WER de 8,5 % à 10,5 %. Le lot sert à remplir un GPU ; sur CPU,
CTranslate2 sature déjà les cœurs avec une seule séquence.

## 1.0.0 — 6 août 2026

Première version publique.

- **Sous-titrage en direct** : moteur whisper.cpp (Vulkan / Metal / CPU), modes
  FR→FR, FR→EN et Auto→EN, presets de latence, LocalAgreement-2 (aucun mot non
  validé n'est affiché), détecteur de parole, lexique de la conférence,
  anti-répétition (`core/textguard.py`), débit de lecture constant (BBC Subtitle
  Guidelines / EBU-TT-D), second sous-titre simultané.
- **Sorties** : fenêtre à fond vert (ou magenta) pour la capture OBS, texte
  poussé par WebSocket vers une source OBS, page web locale.
- **Transcription de fichiers** audio/vidéo (libsndfile, ffmpeg ou PyAV) vers
  `txt`, `md`, `srt`, `vtt`, `json`, `csv`, `tsv`, `lrc` — en lot, un seul
  chargement de modèle, découpage aux silences.
- **Modèles** : 18 variantes multilingues du dépôt ggml officiel, gestionnaire
  avec téléchargement validé (magie ggml + taille) et import manuel, benchmark
  guidé (RTF + WER) avec recommandation en un clic, préchauffage du moteur dès
  l'ouverture, décodage dans un sous-processus dédié.
- **Binaires** Windows x64 (`.exe` onefile), Linux x86_64 (AppImage) et macOS
  arm64 (`.app`) — non signés, voir le README pour le premier lancement.
