"""Anonymise every photo in a folder with one Anonymizer, and write a CSV with one row for each file.

Run it from the repository root:

    python3 examples/batch_photos.py [FOLDER [OUTPUT]]

FOLDER defaults to demo and OUTPUT to anonymized. The blurred photos are written into OUTPUT with the names of the
originals, and so is summary.csv. The originals are only read. A file whose output already exists is skipped, so
the script can be run again after a failure. Files that are not photos (videos, HEIC) are left out or refused, and
the rest of the folder goes on.
"""

import csv
import os
import sys
import time

import anonymizer


def anonymise_folder(folder, output):
    os.makedirs(output, exist_ok=True)
    rows = []
    with anonymizer.Anonymizer() as anon:
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() not in anonymizer.PHOTO_EXTENSIONS:
                continue
            target = os.path.join(output, name)
            row = {
                "file": name,
                "face_items": "",
                "plate_items": "",
                "doubtful": "",
                "left_visible": "",
                "seconds": "",
                "note": "",
            }
            if os.path.exists(target):
                row["note"] = "skipped: the output exists"
            else:
                started = time.perf_counter()
                try:
                    analysis = anon.analyze(os.path.abspath(os.path.join(folder, name)))
                    result = anon.render(analysis, target)
                except anonymizer.ModelLoadError:
                    raise  # the models are missing: every file would fail
                except anonymizer.AnonymizerError as error:
                    row["note"] = error.code
                else:
                    kinds = [item.kind for item in result.report.rows]
                    row.update(
                        face_items=kinds.count("face"),
                        plate_items=kinds.count("plate"),
                        doubtful=result.report.summary()["doubtful"],
                        left_visible=result.visible_check,
                        seconds=round(time.perf_counter() - started, 1),
                        note="; ".join(result.warnings),
                    )
            rows.append(row)
            print(name, "->", row["note"] or "done")
    with open(os.path.join(output, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "demo"
    output = sys.argv[2] if len(sys.argv) > 2 else "anonymized"
    rows = anonymise_folder(folder, output)
    print(len(rows), "photos listed in", os.path.join(output, "summary.csv"))
