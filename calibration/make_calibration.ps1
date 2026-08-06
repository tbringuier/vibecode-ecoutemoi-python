# Generate FR/EN calibration WAVs (48 kHz, 16-bit, mono) with Windows SAPI voices.
# Texts = the app's reference calibration texts (defined in ecoutemoi/constants.py).
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    48000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$textFr = "Bonjour à toutes et à tous, et bienvenue dans cette conférence. Aujourd'hui, nous allons parler de technologie, de réseaux et de stockage distribué. Trois serveurs, douze disques, quarante-deux téraoctets : les chiffres comptent autant que les idées. Merci de votre attention, et place à la démonstration."
$textEn = "Good morning everyone, and welcome to this conference. Today we will talk about technology, networks, and distributed storage. Three servers, twelve disks, and forty-two terabytes: numbers matter as much as ideas. Thank you for your attention, and let's begin the demonstration."

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0

$synth.SelectVoice("Microsoft Hortense Desktop")
$synth.SetOutputToWaveFile((Join-Path $dir "calibration_fr.wav"), $fmt)
$synth.Speak($textFr)

$synth.SelectVoice("Microsoft Zira Desktop")
$synth.SetOutputToWaveFile((Join-Path $dir "calibration_en.wav"), $fmt)
$synth.Speak($textEn)

$synth.SetOutputToNull()
$synth.Dispose()
Get-ChildItem $dir -Filter *.wav | ForEach-Object { "{0}  {1:N0} octets" -f $_.Name, $_.Length }
