# Files written

Every file the Python API writes, field by field: the anonymised photo or video, the saved analysis with its thumbnails folder, and the report as CSV and JSON. Read it to write a program that reads these files without importing `anonymizer`, or to know exactly what an export leaves on disk. The original is only ever opened for reading.

| Call | Writes | Where | Replaces a file that exists? |
|---|---|---|---|
| [`Anonymizer.render`](anonymizer.md#anonymizer-render), [`anonymize_image`](functions.md#anonymize_image), [`anonymize_video`](functions.md#anonymize_video) | the anonymised photo or video | `target`, after a hidden partial file in the same folder | never: [`OutputExists`](errors.md#outputexists) |
| [`compare_video`](functions.md#compare_video) | the side-by-side video, which shows the original unblurred: never publish it | `target` | never: [`OutputExists`](errors.md#outputexists) |
| [`Analysis.save`](anonymizer.md#analysis-save) | the saved analysis (JSON) and its thumbnails folder | `path`, and `path` without `.json` plus `.thumbs` | yes: both are replaced |
| [`Report.to_csv`](data-types.md#report-to_csv), [`Report.to_json`](data-types.md#report-to_json) | the report | `path` | yes |

The blocks of this page run from the repository root on the sample photo `demo/street-westminster.jpg` ([Licence and credits](../licence-and-credits.md)). The first writes every file the others read, into a folder it creates.

```python
import os

import anonymizer

os.makedirs("anonymized", exist_ok=True)
with anonymizer.Anonymizer() as anon:
    analysis = anon.analyze(os.path.abspath("demo/street-westminster.jpg"))
    analysis.save("anonymized/westminster.analysis.json")
    result = anon.render(analysis, "anonymized/westminster.jpg")
result.report.to_csv("anonymized/westminster.csv")
result.report.to_json("anonymized/westminster.json")
print(sorted(os.listdir("anonymized")))
```

```text
# output
['westminster.analysis.json', 'westminster.analysis.thumbs', 'westminster.csv', 'westminster.jpg', 'westminster.json']
```

## The anonymised photo or video

`render` writes the output to a hidden partial file in the folder of `target`, named `.<name>.anonymizer-partial`, checks it, and renames it to `target` only when it is whole. If the call fails, is stopped, or the original changes meanwhile, the partial file is deleted. A process killed outright (`kill -9`, a power cut) can leave it; the next render of the same target replaces it. The folder of `target` must exist, and `target` must not.

A **photo** is written in the format of the original, at its resolution, whatever the name of `target` says. It keeps its pixel size, resolution, orientation, EXIF version, colour space and colour profile, and the GPS block only when `Options.remove_gps` is False; maker notes (which can hold a preview of the unblurred picture), serial numbers, owner, comments, XMP, IPTC and every other piece of metadata go. A JPEG with nothing to blur keeps its image data byte for byte. PNG, TIFF and lossless WebP are written again with the same pixels. A lossy WebP or an AVIF is saved again when it holds more than the list above, with the warning `photo.warning.resaved_for_metadata`. Every photo is checked from its bytes before it gets its name, and one that still holds what must be removed is not delivered (`error.photo.metadata_not_removed`, `error.photo.gps_not_removed`). Padding bytes hidden inside the last scan of a JPEG cannot be told from picture data, and they are not removed without compressing the photo again.

A **video** is always an MP4 with 8-bit H.264 video. It keeps the length, the frame count, the frame rate and the audio (copied, or re-encoded when it cannot be copied); a rotation flag is applied to the picture. The container's metadata, the location a phone writes included, is not carried over, nor are subtitle, timecode and data tracks. Only the first video track is anonymised. The video is decoded again in full and compared before it gets its name.

## The saved analysis

[`Analysis.save`](anonymizer.md#analysis-save) writes the analysis as JSON through a temporary file that replaces `path` at the end, so a stop halfway leaves the earlier file. [`load_analysis`](functions.md#load_analysis) reads it back. The keys come in this order:

| Key | Holds |
|---|---|
| `version` | 3, the version of this file format (not of the Python API). Another version is refused with `error.analysis.version`; a 2.1 analysis is already version 3 and loads |
| `source` | The path of the original **as it was given to `analyze`**. A relative path is looked up from the current folder at render, so analyse an absolute path when the file will be loaded from somewhere else |
| `media` | `"photo"` or `"video"` |
| `fingerprint` | The original's size, modification time and a hash of its content. `render` compares it and refuses a changed original with [`SourceChanged`](errors.md#sourcechanged) |
| `frames`, `fps`, `width`, `height` | The number of frames (1 for a photo), the frame rate (0.0 for a photo), and the size as displayed |
| `options` | What the analysis was made with: `faces`, `plates`, `thoroughness`, `detection`, `detection_ran`, `heavy_only`, `unsure`, `max_zones`, `plate_path`, `vehicle_model`, `plate_band`, and `models`, the name and SHA-256 of every heavy or text model loaded. Not the blur style, strength, margin, fit, gap filling or GPS choice |
| `second` | The second look's counts: `zones_visited`, `zones_skipped`, `confirmed`, `added`, `vehicles_too_small`, `bands_added`, `band_area_share` |
| `items` | One object per item, with the fields of [`Item`](data-types.md#item), in time order |
| `areas` | Each added area by its number, as `{"frame_from", "frame_to", "box": [x0, y0, x1, y1], "follow"}` |
| `zones`, `photo_specs` (a photo) and `video` (a video) | The engine's working data. They are not a contract: read `items` and `second`, which are, and leave these alone |

Editing the file by hand changes what `render` writes. Decide through [`Analysis.set_blur`](anonymizer.md#analysis-set_blur) and [`Analysis.add_area`](anonymizer.md#analysis-add_area), then save again.

`save` also writes a crop of every detected item, `<item_id>.jpg`, in a folder named after the file: the path without `.json`, plus `.thumbs`. The folder is replaced whole, never merged. Deleting it costs the thumbnails and nothing else.

The file is plain JSON, so any program can read it:

```python
import json

with open("anonymized/westminster.analysis.json", encoding="utf-8") as f:
    saved = json.load(f)

print(list(saved))
print(saved["version"], saved["media"], saved["width"], saved["height"])
print(saved["options"]["detection_ran"], [m["name"] for m in saved["options"]["models"]])
blurred = [item for item in saved["items"] if item["blur"]]
print(len(saved["items"]), "items,", len(blurred), "blurred")
```

```text
# output
['version', 'source', 'media', 'fingerprint', 'frames', 'fps', 'width', 'height', 'options', 'second', 'items', 'areas', 'zones', 'photo_specs']
3 photo 960 720
careful ['PP-OCRv6_det_small', 'retinaface-r34']
26 items, 22 blurred
```

## The report

[`Report.to_csv`](data-types.md#report-to_csv) and [`Report.to_json`](data-types.md#report-to_json) write the privacy report, one row for each item, the items left visible and the added areas included. `Result.report` and [`Analysis.report`](anonymizer.md#analysis-report) give the [`Report`](data-types.md#report). It lists what was found and what was done, not what the detectors missed.

The CSV has one column, and each row of the JSON one key, for each field of [`ReportRow`](data-types.md#reportrow), in this order: `clip`, `item_id`, `kind`, `verdict`, `blurred`, `reason`, `first_s`, `last_s`, `frames`, `max_score`, `found_by`, `decision`, `decided_by` and `plate_like`. `item_id` is the engine's number and can skip numbers, where the PDF report counts 1 to n in time order; `plate_like` is empty in the CSV, and `null` in the JSON, for every item that is not a plate found by text.

**The CSV** is UTF-8 with a byte-order mark, so Excel reads accents when the file is opened by a double click; read it in Python with `encoding="utf-8-sig"`. Lines end with a carriage return and a line feed, and a field with a comma, a semicolon, a tab, a quote or a line break is quoted. A text that starts with `=`, `+`, `-`, `@`, a tab or a carriage return gets a `'` in front, so a spreadsheet does not run it. Booleans are `True` and `False`, and a missing value is an empty field.

```python
import csv

with open("anonymized/westminster.csv", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print(list(rows[0]))
plate = next(row for row in rows if row["kind"] == "plate")
print(plate)
doubtful = [row for row in rows if row["verdict"] == "check"]
print(len(rows), "rows,", len(doubtful), "doubtful,", sum(row["blurred"] == "True" for row in doubtful), "of them blurred")
```

```text
# output
['clip', 'item_id', 'kind', 'verdict', 'blurred', 'reason', 'first_s', 'last_s', 'frames', 'max_score', 'found_by', 'decision', 'decided_by', 'plate_like']
{'clip': 'street-westminster.jpg', 'item_id': '5', 'kind': 'plate', 'verdict': 'check', 'blurred': 'False', 'reason': 'plate.reason.vehicle_text', 'first_s': '0.0', 'last_s': '0.0', 'frames': '1', 'max_score': '0.763', 'found_by': 'light', 'decision': 'left visible', 'decided_by': 'engine', 'plate_like': 'False'}
26 rows, 17 doubtful, 13 of them blurred
```

**The JSON** is an object with `version` (2, the version of the report file), `options` (the same object as in the saved analysis), `summary` (the counts of [`Report.summary`](data-types.md#report-summary)) and `items` (the rows). Booleans are `true` and `false`.

## What the Studio adds

The Studio uses the same calls, so everything above holds for its exports. It also writes its settings in `settings.json` ([The Studio](studio.md#settings-languages-and-demo-mode)), and a work folder in the system's temporary folder with copies of crops and uploaded files, deleted when it stops. Its exports are named by the pattern `{name}_anonymized`, with `-2`, `-3` for a name that exists. Its reports sit next to each output: the output's name without its extension, then `.report.pdf`, `.report.csv` or `.report.json`. When two outputs differ only in their extension, the later report keeps it (`x_anonymized.tif.report.csv`). The PDF and JSON reports say when a comparison video was written; the CSV does not.
