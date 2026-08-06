# Journal des versions

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
