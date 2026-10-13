# Progress, cancel and errors

A job sends events while it runs, can be stopped from another thread, and ends with an error that says whether it concerns one file or the whole run. This page shows all three, for scripts and programs that run without a person watching. The same steps, with a command line, are `examples/progress_and_cancel.py` and `examples/handle_errors.py`.

## Read the events

Every call that does work takes `on_event`, a function that receives typed events: `Started`, `Progress`, `ItemFound`, `Analyzed`, `Rendered`, `JobWarning` and `Finished`. The handler runs in the thread that runs the call, so keep it short; an exception in it stops the call. To hand events to another thread, put them in a queue.

This block anonymises the sample photo with a handler that records each kind of event as it arrives, once for each run of the same kind. It writes `anonymized/westminster-events.jpg` under the current folder.

```python
import os

import anonymizer

os.makedirs("anonymized", exist_ok=True)
events = []


def on_event(event):
    name = type(event).__name__
    if isinstance(event, anonymizer.Finished):
        name = f"Finished({event.status})"
    if not events or events[-1] != name:
        events.append(name)


result = anonymizer.anonymize_image("demo/street-westminster.jpg", "anonymized/westminster-events.jpg", on_event=on_event)
print(", ".join(events))
```

```text
# output
Started, Progress, ItemFound, Analyzed, JobWarning, Rendered, Finished(done)
```

`Progress` has a `phase` (`"analyze"`, `"render"` or `"compare"`) and a `fraction` from 0 to 1 that never goes back within a phase. `ItemFound` arrives once for each item, after the analysis and not live. `JobWarning` is sent for each warning of the result, and `Finished` has the `status` `"done"` or `"stopped"`. `event_json` turns an event into a dict for JSON. The fields of each event are on [Data types and events](../reference/data-types.md#event).

## Cancel from another thread

A `CancelToken` is the one object meant to be shared between threads. Pass it as `cancel` and call `token.cancel()` from anywhere: the job stops at the next frame or two. This block cancels a video's analysis after three seconds.

<!-- docs-test: real-only - it cancels a real analysis, which needs a clip that takes time -->
```python
import threading

events.clear()
token = anonymizer.CancelToken()
threading.Timer(3.0, token.cancel).start()
os.makedirs("stopped", exist_ok=True)

try:
    anonymizer.anonymize_video(
        "examples/data/street-madrid-5s.mp4", "stopped/never.mp4", on_event=on_event, cancel=token
    )
except anonymizer.Cancelled:
    events.append("Cancelled raised")

print(", ".join(events))
print("files in stopped/:", os.listdir("stopped"))
```

```text
# output
Started, Finished(stopped), Cancelled raised
files in stopped/: []
```

The stop arrives twice: first the event `Finished("stopped")`, for a display, then the exception `Cancelled`, for the control flow. Nothing is left under the output name. A token stays cancelled for ever, so make a new one for each job; `pause()` and `resume()` hold the work between frames. A photo cannot be broken off in its first pass: `anonymize_image` has no `cancel` argument, and `Anonymizer.analyze` honours one on a photo only between the heavy calls of the second look.

## Errors: one file or the whole run

The Python API's own errors are `AnonymizerError`s with a `code`, and most concern one file: `UnsupportedMedia` (not a photo or video it processes, empty, cut short, or damaged), `OutputExists`, `SourceChanged`, `IncompleteAnalysis`, `Cancelled`, and `AnonymizerError` itself for a missing source or an output that failed its final check. Record those for the file and go on.

Two errors end a run, because every file would fail the same way: `ModelLoadError` (a model is missing or changed) and `ValueError`, a bad option, which is not an `AnonymizerError`. `ModelLoadError` is a subclass of `AnonymizerError`, so catch it first. Some errors are Python's own, such as `FileNotFoundError` when the folder of an output is missing. The full map is [Which error to catch](../reference/index.md#which-error-to-catch).

This block makes files that fail in `errors-demo/` under the current folder and prints the class and the code of each outcome.

```python
import shutil

folder = "errors-demo"
os.makedirs(folder)
photo = os.path.join(folder, "photo.jpg")
shutil.copy("demo/street-westminster.jpg", photo)
with open(os.path.join(folder, "scan.heic"), "wb") as f:
    f.write(b"this is not a photo")
with open("demo/street-westminster.jpg", "rb") as f:
    cut = f.read(20000)
with open(os.path.join(folder, "cut.jpg"), "wb") as f:
    f.write(cut)

options = anonymizer.Options(detection="fast")
outcomes = {}


def attempt(label, call):
    try:
        call()
    except Exception as error:  # here, to see the class and the code
        outcomes[label] = (type(error).__name__, getattr(error, "code", None))
    else:
        outcomes[label] = ("no error", None)


with anonymizer.Anonymizer(options) as anon:
    analysis = anon.analyze(os.path.abspath(photo))
    anon.render(analysis, os.path.join(folder, "out.jpg"))
    attempt("same name again", lambda: anon.render(analysis, os.path.join(folder, "out.jpg")))
    attempt("folder missing", lambda: anon.render(analysis, os.path.join(folder, "no-such-folder", "x.jpg")))
    attempt("not a photo", lambda: anon.analyze(os.path.join(folder, "scan.heic")))
    attempt("cut short", lambda: anon.analyze(os.path.join(folder, "cut.jpg")))
    attempt("source missing", lambda: anon.analyze(os.path.join(folder, "missing.jpg")))
    with open(photo, "ab") as f:
        f.write(b"changed")
    attempt("source changed", lambda: anon.render(analysis, os.path.join(folder, "late.jpg")))
attempt("model folder missing", lambda: anonymizer.Anonymizer(options, models_dir="none").analyze(os.path.abspath(photo)))
attempt("bad option", lambda: anonymizer.Options(strength=0.5))

for label, (name, code) in outcomes.items():
    print(f"{label:22s} {name} {code or ''}".rstrip())
```

```text
# output
same name again        OutputExists error.output.exists
folder missing         FileNotFoundError
not a photo            UnsupportedMedia error.media.unsupported
cut short              UnsupportedMedia error.media.unsupported
source missing         AnonymizerError error.source.missing
source changed         SourceChanged error.source.changed
model folder missing   ModelLoadError error.model.load
bad option             ValueError
```

A photo cut short is `error.media.unsupported` at `analyze`, like a file that is no photo at all; `error.media.damaged` is for a file that decoded at first and stopped later, such as a video partway through. Catch the class, and branch on `.code` only if you must tell them apart. In a command-line tool, give each cause its own exit code, and put `except anonymizer.UnsupportedMedia` before `except anonymizer.AnonymizerError`, because the first matching clause wins: `examples/handle_errors.py` does it.
