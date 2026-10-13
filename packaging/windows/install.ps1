# Sets up Anonymizer Studio from this folder on Windows. NOT TESTED ON WINDOWS: see README.md next to this file.
#   powershell -ExecutionPolicy Bypass -File packaging\windows\install.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

# Python 3.10 or later, from python.org or the Microsoft Store.
& py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10 or later is needed (the 'py' launcher did not find one)." }

& py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r packaging\windows\requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .

# The detection models are not in the repository (MODELS.md: what they are, their licences and checksums): fetch them and check each one.
& .\.venv\Scripts\python.exe scripts\fetch_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Some models could not be fetched into $root\models. Run scripts\fetch_models.py again, or put them there by hand, or set ANONYMIZER_MODELS."
}

# Optional: a native window instead of a browser tab.
#   .\.venv\Scripts\python.exe -m pip install pywebview
Write-Host "Done. Start it with packaging\windows\anonymizer-studio.bat"
