# Reference

The reference of the Python API of Anonymizer Studio (`import anonymizer`), the files it writes and the commands you run. It describes exactly what is there. To learn the Python API by doing, start with [Photos in Python](../tutorials/photos-in-python.md). Coding assistants start at `AGENTS.md` in the repository root.

## What is public

The names in `anonymizer.__all__` are the public API. Everything else in the package is internal and may change: `anonymizer.engine`, `anonymizer.classes`, `anonymizer.contract`, `anonymizer.ui` and the routes the Studio's page talks to. `from anonymizer import *` gives exactly the public names. `anonymizer.contract.ModelLoadError` is not [`anonymizer.ModelLoadError`](errors.md#modelloaderror): catching the internal one catches nothing that a public call raises.

Each entry gives the signature with the annotations of the code, a summary, the parameters with their type, default and valid values, what it returns and raises, and links to related names. Frames count from 0, and a photo is frame 0. A box is `(x0, y0, x1, y1)` in pixels of the original, from the top left. The examples run from the repository root on the sample files in `demo/` and `examples/data/`.

| Page | What is in it |
|---|---|
| [Functions](functions.md) | `anonymize_image`, `anonymize_video`, `compare_video`, `media_info`, `frame_jpeg`, `thumbnail_jpeg`, `load_analysis`, `event_json` and the extension constants |
| [Anonymizer and Analysis](anonymizer.md) | The class that loads the detectors once (`analyze`, `render`, `estimate`), and what `analyze` returns, with the calls that decide (`set_blur`, `add_area`, `preview`, `boxes`, `save`) |
| [Options and CancelToken](options.md) | Every setting, what it changes and when it is read; stopping a job from another thread |
| [Data types and events](data-types.md) | `Item`, `Report`, `ReportRow`, `Result`, `BoxInfo`, `MediaInfo`, `Estimate`, and the seven events |
| [Errors and codes](errors.md) | `AnonymizerError` and its six subclasses, and every reason, warning and error key with its English sentence |
| [Files written](files-written.md) | The output, the saved analysis and the report, field by field |
| [Command line](command-line.md) | The Studio's start command, the model scripts and the environment variables |
| [The Studio](studio.md) | The four screens, their controls, keys and settings |

The pages from Functions to Errors and codes are generated from the docstrings of the code by `scripts/api_docs.py`, so they cannot drift from it.

## Which error to catch

| You catch | It means |
|---|---|
| [`UnsupportedMedia`](errors.md#unsupportedmedia) | The file is not a photo or a video the API takes, is damaged, or stops decoding partway |
| [`ModelLoadError`](errors.md#modelloaderror) | A detection model is missing, changed or could not run, at the first call that needs it |
| [`OutputExists`](errors.md#outputexists) | The target exists. `render` and `compare_video` check first; `anonymize_*` find it after the analysis |
| [`SourceChanged`](errors.md#sourcechanged) | The original is no longer the file that was analysed |
| [`IncompleteAnalysis`](errors.md#incompleteanalysis) | Part of the file could not be checked, so it is not delivered as clean |
| [`Cancelled`](errors.md#cancelled) | The job was stopped through its `CancelToken` |
| [`AnonymizerError`](errors.md#anonymizererror) | Another code, such as `error.source.missing`, `error.analysis.version` or `error.output.check_failed`: the class lists them all |
| `FileNotFoundError`, `NotADirectoryError` | The folder of a target is missing or is a file (checked before a long analysis), a file to read is missing, or a loaded analysis names a relative source not found from the current folder |
| `ValueError` | A value is not valid: an `Options` field, a frame or a box, a view name, the captions of `compare_video` |
| `KeyError` | No item has that number (`Analysis.item`, `set_blur`, `remove_area`) |
| `json.JSONDecodeError` | `load_analysis` was given a file that is not JSON |
| `dataclasses.FrozenInstanceError` | You assigned to a field of `Options` |

In a batch, let `ModelLoadError` and `ValueError` end the run, and record any other `AnonymizerError` for its file and go on. Put `except anonymizer.ModelLoadError: raise` before `except anonymizer.AnonymizerError`, because `ModelLoadError` is a subclass.

## Threads and processes

- An `Anonymizer` is not safe to share between threads: give each thread its own. [`CancelToken`](options.md#canceltoken) is the one object meant to be used from another thread.
- `analyze` and `render` run in the thread that calls them, and so does an event handler; an exception in the handler comes out of the call.
- ONNX Runtime uses the threads you set with `Anonymizer(threads=...)`; 0 lets the runtime decide.
- The Python API runs in your process and starts no server. The Studio runs the engine in a worker process. The only network use of the product is [`scripts/fetch_models.py`](command-line.md#scriptsfetch_modelspy), once.
