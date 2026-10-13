# Anonymizer Studio: notes for AI coding assistants

Read this file before you write code that calls Anonymizer Studio: it is enough for the common tasks. Every statement is
checked against version 2.2.0, and every block runs from the repository root on its sample files in `demo/` and
`examples/data/`; elsewhere, use your own files. The full reference starts at `docs/reference/index.md`.

Anonymizer Studio blurs faces and licence plates in photos and videos, offline, on this machine. It is a Python API
(`import anonymizer`) and a local web application (the Studio). Python 3.10 or newer; tried on Linux only. By default it blurs
the plate itself and the face itself, with a tight margin and a soft edge. What the detectors are not sure about (a doubtful
item) is listed and blurred only when it is placed like a plate or a face. **A plate or a face the detectors do not find is
neither blurred nor listed.** The only network use is `scripts/fetch_models.py`, once. `import anonymizer` gives every public
name and loads no model; never import `anonymizer.engine`, `.classes`, `.contract` or `.ui`, which are not an API
(`anonymizer.contract.ModelLoadError` is not `anonymizer.ModelLoadError`).

Three things the first run shows. A street photo often lists a dozen or more doubtful items (17 of 26 for
`demo/street-westminster.jpg`): the default rule blurs most, and `visible_check` counts the ones left visible (4 there). A
photo takes seconds; the first call also loads the models. HEIC, RAW, GIF, text files, folders and photos
cut short are refused at `analyze` with `UnsupportedMedia` (`error.media.unsupported`).

## Common tasks

Each block is a whole script: copy one and change the paths. A rendered file is never overwritten: only tasks 3 and 4 delete it first, so the
others raise `OutputExists` the second time; `save`, `to_csv` and `to_json` replace theirs, the `.analysis.thumbs/` folder included.

**1. One photo or one video.** The folder of the target must exist. Give a photo the extension of the original, and a video `.mp4`.
Read `result.visible_check`: with the defaults, doubtful items that are not placed stay visible.

```python
import os

import anonymizer

os.makedirs("out", exist_ok=True)
result = anonymizer.anonymize_image("demo/street-westminster.jpg", "out/westminster.jpg")
print(result.visible_check, result.warnings)
video = anonymizer.anonymize_video("examples/data/street-madrid-5s.mp4", "out/madrid.mp4")
```
```text
# output
4 ['warning.review.check_left_visible']
```

**2. A folder of photos to a CSV.** One `Anonymizer` for the whole folder, and `PHOTO_EXTENSIONS` to skip what is not a
photo (`demo/` also holds videos). Faces and plates are the items of that `kind`, doubtful ones included; `to_check` counts
the items whose `verdict` is `"check"`. A damaged photo still raises `UnsupportedMedia`: task 3 has the `except` clauses.

```python
import csv
import glob
import os
import time

import anonymizer

os.makedirs("out", exist_ok=True)
with anonymizer.Anonymizer() as anon, open("out/summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "faces", "plates", "to_check", "seconds"])
    for path in sorted(glob.glob("demo/*")):
        if os.path.splitext(path)[1].lower() not in anonymizer.PHOTO_EXTENSIONS:
            continue  # not a photo
        start = time.perf_counter()
        analysis = anon.analyze(os.path.abspath(path))
        seconds = round(time.perf_counter() - start, 2)
        items = analysis.items
        faces = sum(item.kind == "face" for item in items)
        plates = sum(item.kind == "plate" for item in items)
        to_check = sum(item.verdict == "check" for item in items)
        writer.writerow([os.path.basename(path), faces, plates, to_check, seconds])
```

**3. A folder with files that cannot be processed: no crash, each error's code, an exit code.** A text file, an empty or
damaged photo or video, and a folder are all `UnsupportedMedia` with the code `error.media.unsupported` already at `analyze`.
`ModelLoadError` is a subclass of `AnonymizerError`: it goes first, and it ends the run.

```python
import glob
import os
import shutil
from pathlib import Path

import anonymizer

shutil.rmtree("out/mixed", ignore_errors=True)
shutil.rmtree("out/mixed-anonymized", ignore_errors=True)
os.makedirs("out/mixed")
os.makedirs("out/mixed-anonymized")
shutil.copy("demo/street-westminster.jpg", "out/mixed/crossing.jpg")
Path("out/mixed/notes.txt").write_text("not a photo")
Path("out/mixed/broken.jpg").write_bytes(Path("demo/street-westminster.jpg").read_bytes()[:2000])  # cut short

failed = 0
with anonymizer.Anonymizer() as anon:
    for path in sorted(glob.glob("out/mixed/*")):
        name = os.path.basename(path)
        try:
            analysis = anon.analyze(os.path.abspath(path))
            anon.render(analysis, os.path.join("out/mixed-anonymized", name))
        except anonymizer.ModelLoadError:
            raise  # every file would fail: stop the run
        except anonymizer.AnonymizerError as error:
            failed += 1
            print(name, error.code)  # this file only; go on with the next
print("exit", 2 if failed else 0)  # in a script: sys.exit(2 if failed else 0)
```
```text
# output
broken.jpg error.media.unsupported
notes.txt error.media.unsupported
exit 2
```

**4. A video: analyse, list the doubtful items, decide one, render, save, reload.** `analyze` writes nothing. A decision
lives in the `Analysis`. Seconds are `frame / analysis.fps`. Analyse an absolute path when the analysis will be loaded from
another folder.

```python
import os

import anonymizer

os.makedirs("out", exist_ok=True)
for old in ("out/review.mp4", "out/review.report.json"):
    if os.path.exists(old):
        os.remove(old)
with anonymizer.Anonymizer() as anon:
    analysis = anon.analyze(os.path.abspath("examples/data/street-madrid-5s.mp4"))
    doubtful = [item for item in analysis.items if item.verdict == "check"]
    for item in doubtful:
        first_s, last_s = item.first_frame / analysis.fps, item.last_frame / analysis.fps
        print(item.item_id, item.kind, round(first_s, 1), round(last_s, 1), item.reason)
    if doubtful:
        item_id = doubtful[0].item_id
        analysis.set_blur(item_id, True)  # confirm the doubtful item: blur it
    result = anon.render(analysis, "out/review.mp4")
    analysis.save("out/review.analysis.json")  # replaces an earlier file; also writes the crops in out/review.analysis.thumbs/
    result.report.to_json("out/review.report.json")
if doubtful:
    print(item_id, anonymizer.load_analysis("out/review.analysis.json").item(item_id).blur)
```
```text
# output
2 face 0.1 0.5 face.reason.low_verify
3 face 0.5 0.6 face.reason.low_verify
6 face 1.8 4.0 face.reason.second_found
2 True
```

**5. Progress per phase, then a cancel from another thread.** Handlers run in the calling thread. A video's `analyze` and
`render` phases each end with one `Progress` whose `fraction` is exactly 1.0; a stopped job never gets there. A cancelled
job sends `Finished("stopped")` and then raises `Cancelled`. A photo's first pass cannot be broken off: cancel a video.

```python
import os
import threading

import anonymizer

os.makedirs("out", exist_ok=True)

def show(event):
    if isinstance(event, anonymizer.Progress) and event.fraction >= 1.0:
        print(event.phase, "100%")
    elif isinstance(event, anonymizer.Finished):
        print("finished", event.status)

anonymizer.anonymize_video("examples/data/street-madrid-5s.mp4", "out/progress.mp4", on_event=show)
print("done")
token = anonymizer.CancelToken()
threading.Timer(1.0, token.cancel).start()
try:
    anonymizer.anonymize_video("examples/data/street-madrid-5s.mp4", "out/second.mp4", on_event=show, cancel=token)
except anonymizer.Cancelled:
    print("stopped")
```
```text
# output
analyze 100%
render 100%
finished done
done
finished stopped
stopped
```

**6. GPS before and after.** A photo's GPS block is dropped by default, and so is all other metadata but a short list
(orientation, resolution, colour space and profile). Keep the location with `Options(remove_gps=False)`. A video's output
never carries the container's metadata.

```python
import os

from PIL import Image

import anonymizer

os.makedirs("out", exist_ok=True)
picture = Image.open("demo/street-westminster.jpg")  # a copy with an invented location, as a phone writes it
exif = picture.getexif()
gps = exif.get_ifd(0x8825)
gps[1], gps[2], gps[3], gps[4] = "N", (0.0, 0.0, 0.0), "W", (30.0, 0.0, 0.0)
picture.save("out/holiday.jpg", exif=exif, quality=95)

print("original", anonymizer.media_info("out/holiday.jpg").has_gps)
result = anonymizer.anonymize_image("out/holiday.jpg", "out/holiday-default.jpg", anonymizer.Options(detection="fast"))
print("default", anonymizer.media_info(result.output).has_gps)
options = anonymizer.Options(detection="fast", remove_gps=False)
result = anonymizer.anonymize_image("out/holiday.jpg", "out/holiday-kept.jpg", options)
print("kept", anonymizer.media_info(result.output).has_gps)
```
```text
# output
original True
default False
kept True
```

**7. Read the saved analysis and the report without importing `anonymizer`.** Both are plain JSON; the report is also
CSV (`utf-8-sig`, `True`/`False`). This reads the files task 4 wrote.

```python
import json

with open("out/review.analysis.json", encoding="utf-8") as f:
    saved = json.load(f)
with open("out/review.report.json", encoding="utf-8") as f:
    report = json.load(f)
doubtful_ids = [item["item_id"] for item in saved["items"] if item["verdict"] == "check"]
print(len(saved["items"]), doubtful_ids)
print(report["summary"]["blurred"], report["summary"]["doubtful"])  # report["items"] holds the rows
```
```text
# output
5 [2, 3, 6]
4 3
```

**8. Blur every doubtful item, or decide by size.** For all of them, `Options(unsure="blur")` before the analysis. To keep them
all visible, `Options(unsure="keep")`. A person's `set_blur` always wins.

```python
import os

import anonymizer

os.makedirs("out", exist_ok=True)
with anonymizer.Anonymizer() as anon:
    analysis = anon.analyze(os.path.abspath("demo/street-westminster.jpg"))
    for box in analysis.boxes(0, 0, only_blurred=False)[0]:  # a photo is frame 0
        if analysis.item(box.item_id).verdict == "check" and box.box[2] - box.box[0] >= 30:
            analysis.set_blur(box.item_id, True)  # confirm the doubtful ones 30 pixels wide or more
    result = anon.render(analysis, "out/westminster-by-size.jpg")
    print(result.visible_check)
```
```text
# output
2
```

**9. Which `Options` fields act when.** `faces`, `plates`, `detection`, `plate_path`, `vehicle_model`, `plate_band` and `fit`
are read once, when the detectors load: make a new `Anonymizer` to change them. `thoroughness`, `unsure`, `max_zones`,
`heavy_only` and `gap_fill` are read at each `analyze`. `blur`, `strength`, `extra_margin`, `remove_gps` and `gap_fill` are read
at each `render`: assign `dataclasses.replace(anon.options, blur="pixelate")` to `anon.options`.

**10. The models** are not in the package: `python3 scripts/fetch_models.py [--models FOLDER]` fetches them once. `Anonymizer`
looks in `models_dir`, then `ANONYMIZER_MODELS`, then `./models`, then the source tree's `models`. A missing or changed file is
a `ModelLoadError` at the first `analyze`, not at `Anonymizer()`.

## Fields at a glance

| Object | Fields |
|---|---|
| `Item` (`analysis.items`) | `item_id`, `kind` (`"face"`, `"plate"`, `"manual"`), `verdict` (`"blur"`, or `"check"` for a doubtful item), `reason` (a key such as `face.reason.low_verify`: print the key, no function turns it into a sentence), `first_frame`, `last_frame`, `max_score`, `blur` (bool: blurred now), `found_by`, `decided_by` |
| `Analysis` | `items`, `item(id)`, `media` (`"photo"`, `"video"`), `frames`, `fps` (0.0 for a photo), `width`, `height`, `source`, `set_blur`, `add_area`, `boxes`, `preview`, `report`, `summary`, `save` |
| `Result` | `output`, `report`, `warnings` (keys), `seconds`, `visible_check` (doubtful items left visible) |
| `Progress` | `clip_id`, `phase` (`"analyze"`, `"render"`, `"compare"`), `fraction` (0 to 1), `frame`, `frames`; `Finished`: `clip_id`, `status` (`"done"`, `"stopped"`) |
| `Report.summary()` | `items`, `blurred`, `doubtful`, `blurred_by_default`, `confirmed_by_person`, `turned_off_by_person`, `left_visible_by_default`, then the second look's counts |
| `ReportRow` (CSV, JSON) | `clip`, `item_id`, `kind`, `verdict`, `blurred`, `reason`, `first_s`, `last_s`, `frames`, `max_score`, `found_by`, `decision` (`"blurred"`, `"left visible"`), `decided_by`, `plate_like` |
| `MediaInfo` | `media`, `width`, `height`, `frames`, `fps`, `duration_s`, `codec`, `has_audio`, `has_gps`, `size_bytes` |

The codes a batch meets, each on `error.code`: `UnsupportedMedia`: `error.media.unsupported` (not a photo or video, empty or
damaged, a folder), `error.media.damaged` (raised by `render`, when the original stops decoding partway;
a damaged photo is `error.media.unsupported` at `analyze`), `error.photo.several_images`. `OutputExists`:
`error.output.exists`. `SourceChanged`: `error.source.changed`. `IncompleteAnalysis`: `error.analysis.incomplete`.
`Cancelled`: `error.cancelled`. `ModelLoadError`: `error.model.load` (stop the run). Other `AnonymizerError` codes:
`error.source.missing`, `error.photo.not_enough_memory`, `error.output.check_failed`. Builtins also come out:
`FileNotFoundError` for a missing folder, `ValueError` for a bad option or frame, `KeyError` for an unknown item id.

## The mistakes, in the order a model makes them

1. **A relative path in a saved analysis.** `Analysis.source` is the path as you gave it. `load_analysis("../a.json")` and `render`
   from another folder raise `FileNotFoundError: 'in.mp4'`, not `SourceChanged`. Pass `os.path.abspath(path)` to `analyze`.
2. **The output keeps the format of the original.** A `.jpg` rendered to `out.png` is a JPEG named `out.png`. A video is always an MP4.
3. **The destination folder must exist, and the target must not.** A missing folder is `FileNotFoundError`, raised before the long
   analysis. An existing target is `OutputExists`: pick another name. `anonymize_*` find a taken target only after the analysis,
   `Anonymizer.render` at once.
4. **The default hides little, on purpose.** `demo/street-taipei.jpg` with the defaults blurs nothing: every doubtful item is left
   visible and the file is still written. Read `Result.warnings` and `Result.visible_check`.
5. **A miss is silent.** A face or a plate that is not found is neither blurred nor in the report. Look at the output.
   `Analysis.add_area` covers what is missed. `ReportRow.plate_like` False: not shaped like a plate, listed
   because it may still be one.
6. **A decision lives in the `Analysis`.** `set_blur` and `add_area` change it, `render` uses it, `save` keeps it. Calling `analyze`
   again to apply a decision replaces the decisions.
7. **One `Anonymizer` per thread.** `on_event` runs in the calling thread, and an exception in it aborts the call; hand events
   to another thread with a queue. A `CancelToken` stays cancelled: make a new one for each job.
8. **`Analysis.boxes` and `Analysis.preview` take `options=None`, which means `Options()`**, not the options of your `Anonymizer`.
   Pass `options=anon.options` when you changed the margin, the gap filling or the fit.
9. **The comparison video shows the original, unblurred.** `compare_video` is for checking. Never publish it.

## Never

- Never invent a field, a method or an option. If the reference (`docs/reference/`) does not list it, it does not exist.
- Never call `anonymize_image` or `anonymize_video` in a loop over a folder. Make one `Anonymizer`.
- Never edit a saved analysis by hand: decide through `set_blur` and `add_area`, and save again.
- Never turn a doubtful item off (`set_blur(id, False)`, `Options(unsure="keep")`) on your own authority: an item left visible by
  mistake publishes a face or a plate. Show the person the item (`Analysis.preview`, `Analysis.thumbnail_jpeg`) and apply what
  they decide.
