# Build the pywhispercpp engine wheel with Vulkan for Windows x64.
# Used by the packaging CI job; runnable locally for GPU testing.
#
# Requirements: Visual Studio Build Tools + CMake + Vulkan SDK LunarG
# (VULKAN_SDK env set, glslc dans le PATH).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$WheelDir = if ($env:WHEEL_DIR) { $env:WHEEL_DIR } else { "wheelhouse" }

if (-not $env:VULKAN_SDK) {
    Write-Error "VULKAN_SDK n'est pas défini — installez le Vulkan SDK (LunarG) d'abord."
}

# MSBuild FileTracker (FTK1011) échoue au-delà de MAX_PATH (260) et ignore
# LongPathsEnabled : le TEMP profond de pip (Users\...\AppData\Local\Temp\
# pip-wheel-*\pywhispercpp_<hash>\build\...\*.tlog) dépasse. TEMP court obligatoire.
$ShortTmp = "C:\bw"
New-Item -ItemType Directory -Force -Path $ShortTmp | Out-Null
$env:TMP = $ShortTmp
$env:TEMP = $ShortTmp

# Générateur Ninja obligatoire : sous le générateur Visual Studio, ggml compile
# vulkan-shaders-gen (outil hôte) dans un CMake enfant lancé par MSBuild, qui ne
# retrouve pas cl.exe (« No CMAKE_C_COMPILER could be found »). Avec Ninja +
# l'environnement vcvars (cl dans le PATH), parent et enfant compilent pareil —
# c'est la configuration de la CI Vulkan de llama.cpp/whisper.cpp.
# Le setup.py de pywhispercpp respecte CMAKE_GENERATOR et n'ajoute -A x64 que
# pour les générateurs multi-config.
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Error "cl.exe absent du PATH — initialisez l'environnement MSVC (vcvars64) d'abord."
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    choco install ninja -y --no-progress | Out-Null
}
ninja --version
$env:CMAKE_GENERATOR = "Ninja"

# Pin to the locked version so the artifact matches uv.lock.
$Version = (uv run python -c "import importlib.metadata as m; print(m.version('pywhispercpp'))").Trim()

# GGML_NATIVE=OFF: portable artifact, not runner-tuned. AVX2+FMA+F16C+BMI2 =
# le variant « avx2 » canonique de ggml (Intel Haswell 2013+ / AMD Zen 2017+).
# Pas d'AVX-512 : absent de nombreux CPU grand public, l'artefact ne démarrerait pas.
$env:CMAKE_ARGS = "-DGGML_VULKAN=1 -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON"

# Chemins ultra-courts obligatoires : l'ExternalProject vulkan-shaders-gen de
# ggml empile ~90 caractères (…\vulkan-shaders-gen-prefix\src\…\CMakeScratch\…)
# et crève MAX_PATH même depuis un TEMP court. On extrait le sdist dans C:\s\p
# et pip (>= 21.3) construit IN-TREE : pire chemin objet ~230 < 250.
$SrcRoot = "C:\s"
if (Test-Path $SrcRoot) { Remove-Item -Recurse -Force $SrcRoot }
New-Item -ItemType Directory -Force -Path $SrcRoot | Out-Null
# Chemins longs autorisés au niveau OS en ceinture (cl/ninja sont long-path aware).
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name LongPathsEnabled -Value 1 -ErrorAction SilentlyContinue

Write-Host "Fetching pywhispercpp==$Version sdist"
uv run --with pip python -m pip download "pywhispercpp==$Version" --no-binary :all: --no-deps --no-cache-dir -d $SrcRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
tar -xzf (Get-ChildItem "$SrcRoot\pywhispercpp-*.tar.gz").FullName -C $SrcRoot
Rename-Item "$SrcRoot\pywhispercpp-$Version" "p"

# Patch du setup.py : il hardcode bin\<Release> (convention des générateurs
# multi-config Visual Studio) pour l'étape delvewheel ; sous Ninja
# (single-config) les DLLs sont dans bin\ et la réparation échoue sur
# « Unable to find library: whisper.dll ». Repli sur bin\ si bin\Release est vide.
$SetupPy = "$SrcRoot\p\setup.py"
$PatchOld = "        dll_folder = os.path.join(self.build_temp, '_pywhispercpp', 'bin', cfg)"
$PatchNew = "        dll_folder = os.path.join(self.build_temp, '_pywhispercpp', 'bin', cfg)`n" +
            "        if not list(Path(dll_folder).glob('*.dll')):  # Ninja: single-config, DLLs dans bin\`n" +
            "            dll_folder = os.path.join(self.build_temp, '_pywhispercpp', 'bin')"
$SetupContent = Get-Content $SetupPy -Raw
if (-not $SetupContent.Contains($PatchOld)) {
    Write-Error "setup.py inattendu : point de patch dll_folder introuvable (version pywhispercpp changée ?)"
}
$SetupContent.Replace($PatchOld, $PatchNew) | Set-Content $SetupPy -NoNewline -Encoding UTF8
uv run python -c "import ast; ast.parse(open(r'$SetupPy', encoding='utf-8-sig').read()); print('setup.py patché : syntaxe OK')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building pywhispercpp==$Version from source with: $($env:CMAKE_ARGS)"
# pip >= 21.3 construit les répertoires locaux in-tree : build\ vit sous C:\s\p,
# l'isolation de l'environnement de build n'y change rien.
uv run --with pip python -m pip wheel "$SrcRoot\p" --no-deps --no-cache-dir -w $WheelDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Get-ChildItem $WheelDir
Write-Host "Install into the project venv with:"
Write-Host "  uv pip install $WheelDir\pywhispercpp-*.whl --force-reinstall"
