# FAQ and troubleshooting

The questions people ask first, then the problems people meet, each with its cause and what to do.

## Questions

**Does anything leave my computer?** No file, frame or figure. `scripts/fetch_models.py` downloads the models once, and that is the product's only network use ([Privacy](concepts/privacy.md)).

**Does it change my original?** No. The original is only read, and an output never replaces an existing file.

**How do I know it found every face and plate?** You cannot know from the report, which lists what was found, not what was missed. Play the output and look. [Accuracy and limits](accuracy.md) says how often faces and plates were missed in the tests.

**Why is a plate not blurred?** The detectors did not find it (use **Add missed area**), or it is on something not found as a vehicle, or it is **Not sure** and not placed like a plate (press `B`, or **Blur all doubtful**), or it is in a video, where a doubtful plate is left visible by default. [How it works](concepts/how-it-works.md#doubtful-items) has the rule.

**Why is nothing blurred in this photo?** In dense traffic the detectors often list many text regions that they do not judge to be plates, and the default rule leaves them visible. Look at the **Not sure** items in **Review**. **Blur all doubtful**, or `Options(unsure="blur")` in Python, blurs them all, and some things that are not plates with them.

**What format does the output have?** A photo keeps its format and size. A video always comes out as an MP4 with 8-bit H.264 video, with the length, frame rate and size of the original. A `.mov` leaves as `.mp4`. Only the first video track is processed; the sound is copied, or re-encoded when it cannot be, and the export says so. HEIC, RAW and GIF files are refused: export them as JPEG or TIFF first.

**Is the comparison video safe to publish?** No. Its left half is the original, with every face and plate visible.

**Can I use it without the Studio?** Yes. The Python API does everything the Studio does: start with [Photos in Python](tutorials/photos-in-python.md), or give an AI assistant `AGENTS.md` from the repository root.

## Problems

| You see | Cause and fix |
|---|---|
| `ensurepip is not available`, or `Failed building wheel for UNKNOWN` | Ubuntu's `venv` without its helper, or its old pip. Run `sudo apt install python3-venv` or use the `--user` install, and upgrade pip first ([Getting started](getting-started.md#install)) |
| `anonymizer-studio: command not found` after `pip install --user` | The user script folder is not on `PATH` until your next login. Use `python3 -m anonymizer.ui` |
| `ModuleNotFoundError: numpy` when the Studio starts | It was started with `python3 -s`, which hides a `--user` install. Leave `-s` out |
| "Anonymizer Studio could not start on port N" | A Studio is already running there. Use its address, or `--port` another port |
| "This page is not connected to the Studio" or "This request is not allowed" | A bookmark or an old tab. Open the address the Studio printed, and allow cookies for `127.0.0.1` |
| "The Studio is not running" | It stopped after ten minutes without a page, a closed terminal or `kill`. Start it again and open the new address |
| `ModelLoadError` | A model is missing, unlisted or changed. Run `python3 scripts/fetch_models.py`, or set `ANONYMIZER_MODELS` ([Getting started](getting-started.md#fetch-the-models)) |
| `OutputExists` | The output name is taken. Choose another, or delete the old output |
| `FileNotFoundError`: "The destination folder does not exist" | The Python API does not create it. Make it with `os.makedirs` |
| `UnsupportedMedia` | A format it does not process, or an empty or cut-short file. Convert it, or copy it again from its source |
| `SourceChanged` | The original changed or moved after the analysis. Analyse it again |
| `FileNotFoundError` for the original after `load_analysis` | The analysis keeps a relative path. Analyse an absolute one (`os.path.abspath`) |
| `IncompleteAnalysis`: "Part of this file could not be checked" | A detector failed on part of the file. Analyse it again, and report it if it keeps failing |
| A photo named `.png` is a JPEG | The output keeps the original's format. Use its extension |
| The CSV opens in one column in Excel | A language whose list separator is the semicolon. Use Data > From Text/CSV and choose the comma |
| "A PDF report needs a Chromium-based browser" | No Chrome, Edge or Chromium is installed. Install one; the CSV and JSON do not need it |
| **Stop** takes long on a photo | A photo's first pass cannot be broken off. Wait; a video stops within a frame or two |
| "did not answer for N minutes" | The engine was silent for ten minutes and was stopped, without writing the unfinished file. Free memory and processor time, then export again |
