# Photos in Python

This tutorial anonymises one photo with the Python API, then a folder of photos with a CSV of what was found in each. Install first ([Getting started](../getting-started.md)), and run every block from the repository root. The samples are `demo/street-westminster.jpg`, a photo of a pedestrian crossing in London, and `demo/street-taipei.jpg`, a photo of dense traffic in Taipei ([Licence and credits](../licence-and-credits.md)). Code readers, people or AI assistants, also have `AGENTS.md` in the repository root.

## One photo

`anonymize_image` analyses a photo and writes the anonymised copy in one call. This block writes `anonymized/westminster.jpg` under the current folder.

```python
import os

import anonymizer

os.makedirs("anonymized", exist_ok=True)
result = anonymizer.anonymize_image("demo/street-westminster.jpg", "anonymized/westminster.jpg")
summary = result.report.summary()
print("written:", result.output)
print("items:", summary["items"], "blurred:", summary["blurred"])
print("doubtful items left visible:", result.visible_check)
print("warnings:", list(result.warnings))
```

```text
# output
written: anonymized/westminster.jpg
items: 26 blurred: 22
doubtful items left visible: 4
warnings: ['warning.review.check_left_visible']
```

Read the last two lines before you publish: `visible_check` counts the doubtful items that stay visible under the default rule ([How it works](../concepts/how-it-works.md#doubtful-items)), and they may be plates. The first call also loads the models, so it is slower than the next. A second run of the block raises `OutputExists`: a file is never replaced.

## A folder of photos

For more than one file, make one `Anonymizer`: `anonymize_image` loads the models again at every call. For each file, `analyze` finds the items and `render` writes the output. This block builds a folder `photos/` with the two sample photos and a file that is not a photo, anonymises it into `anonymized/`, and writes `anonymized/summary.csv`.

```python
import csv
import os
import shutil

import anonymizer

os.makedirs("photos", exist_ok=True)
for name in ("street-westminster.jpg", "street-taipei.jpg"):
    shutil.copy(os.path.join("demo", name), "photos")
with open("photos/scan.heic", "wb") as f:
    f.write(b"this is not a photo")
os.makedirs("anonymized", exist_ok=True)

FIELDS = ["file", "face_items", "plate_items", "doubtful", "left_visible", "note"]
rows = []
with anonymizer.Anonymizer() as anon:
    for name in sorted(os.listdir("photos")):
        try:
            analysis = anon.analyze(os.path.abspath(os.path.join("photos", name)))
            result = anon.render(analysis, os.path.join("anonymized", name))
        except anonymizer.ModelLoadError:
            raise  # the models are missing: every file would fail
        except anonymizer.AnonymizerError as error:
            print(f"{name}: not processed ({error.code})")
            rows.append({"file": name, "note": error.code})  # the other columns stay empty
            continue
        kinds = [row.kind for row in result.report.rows]
        rows.append(
            {
                "file": name,
                "face_items": kinds.count("face"),
                "plate_items": kinds.count("plate"),
                "doubtful": result.report.summary()["doubtful"],
                "left_visible": result.visible_check,
                "note": "; ".join(result.warnings),
            }
        )

with open("anonymized/summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
    writer.writeheader()
    writer.writerows(rows)
with open("anonymized/summary.csv", encoding="utf-8") as f:
    print(f.read())
```

```text
# output
scan.heic: not processed (error.media.unsupported)
file,face_items,plate_items,doubtful,left_visible,note
scan.heic,,,,,error.media.unsupported
street-taipei.jpg,0,17,17,17,warning.review.check_left_visible
street-westminster.jpg,24,2,17,4,warning.review.check_left_visible
```

A file that fails raises an `AnonymizerError`: the block records its `code`, writes nothing for it and goes on. Only `ModelLoadError`, a missing model, and `ValueError`, a bad option, end the run, because every file would fail. `scan.heic` is text with a photo's name, so it is `UnsupportedMedia`. The path given to `analyze` is absolute, so a saved analysis still finds the original from another folder. The counts are the report's rows, doubtful items included.

**Read the last two columns before you publish.** None of the Taipei photo's doubtful items is placed like a plate, so nothing in it is blurred, and every call succeeds. `Options(unsure="blur")` blurs every doubtful item instead. To run the folder again, write to a new folder, or skip a photo whose output exists, as `examples/batch_photos.py` does.

Next: [Review a video in code](../how-to/video-review-in-code.md), or [Progress, cancel and errors](../how-to/progress-and-errors.md).
