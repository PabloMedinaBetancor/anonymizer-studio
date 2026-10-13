# Anonymizer Studio on Windows: prepared, not tested

Nobody has run any of this on Windows. It was written on Linux, from what is known about Windows, and has never been
executed there. Treat every line as unverified until someone installs it on a real machine and runs the checklist at the
end.

## What gets installed

- A Python 3.10 or later virtual environment in `.venv\` with `numpy`, `opencv-python-headless`, `onnxruntime`, `av`
  (FFmpeg comes inside its wheel) and `pillow` (`requirements.txt`), and this project installed in it (`pyproject.toml`).
- The detection models in `models\` (or wherever `ANONYMIZER_MODELS` points). They are not in the repository:
  `python scripts\fetch_models.py` downloads them and checks their SHA-256 (never run on Windows); `MODELS.md` says what they
  are, where they come from and their checksums.
- Optional: `pywebview` for a native window. Without it, Studio opens in the default browser.

## How to install and start it

    powershell -ExecutionPolicy Bypass -File packaging\windows\install.ps1
    packaging\windows\anonymizer-studio.bat

`anonymizer-studio.bat` runs `python -m anonymizer.ui`: a local server on `127.0.0.1` (a free port) and a browser tab
that opens on it. Add `--demo` for the hosted-demo limits, `--no-browser` to only print the address, `--engine thread`
to run the engine inside the server instead of in a worker process.

## What it uses from Windows

- Settings are kept in `%APPDATA%\AnonymizerStudio\settings.json`.
- The PDF report is printed by Chrome or Edge, found in `Program Files`; without either, the PDF option is disabled and
  says why. CSV and JSON reports need nothing.
- The engine runs in a child process (`python -m anonymizer.ui.worker`) started without a console window. A worker that
  says nothing for 10 minutes is terminated.
- Nothing is sent over the network. The server listens on `127.0.0.1` only.

## What is known to be missing

- No installer (`.exe` or `.msi`) and no PyInstaller build: only the virtual environment above.
- The native window (`pywebview`) has never been started, on any system.
- Speed on Windows has not been measured, so no number for it appears anywhere in the README.

## Checklist for the first real run (what to verify, in order)

1. `install.ps1` finishes and `.venv\Scripts\python.exe -c "import anonymizer"` works.
2. `python -m unittest discover -s tests -t .` passes (the tests that need the real models are off unless `ANONYMIZER_HEAVY=1`).
3. `anonymizer-studio.bat` opens the page; Import lists the user's folders (drive letters included).
4. A short public clip analyzes, can be reviewed and exports; the exported file plays and the blur is where the boxes were.
5. Stop and Pause act within a couple of seconds; killing the worker from Task Manager gives an error in the page, not a
   dead server.
6. The PDF report is made with Chrome and with Edge, and the option is disabled with neither installed.
