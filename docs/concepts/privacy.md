# Privacy

What leaves your computer, what Anonymizer Studio promises about your files, and what it does not: read it before you let it touch your footage.

## Nothing leaves the computer

The Studio and the Python API open no network connection, and a test watches for one. The only network use is `scripts/fetch_models.py`, which downloads the models once. The Studio listens on `127.0.0.1` only and loads nothing from outside; its work folder is deleted when it stops, and the PDF report is printed by a local browser with its network cut.

## What is promised about your files

- **Originals are only read**, and **an output never replaces a file**: the Python API raises `OutputExists`, the Studio adds `-2`. Only `Report.to_csv` and `Report.to_json` replace a file of the same name.
- **A half-written file is never under the final name**: the output is written under a hidden name, checked, then renamed. Only a process killed outright can leave the hidden file.
- **A file that changed is not exported** (`SourceChanged`), and **a frame or photo the detectors failed on is never exported as clean** (`IncompleteAnalysis`).
- **A photo keeps only its resolution, orientation, EXIF version, colour space, pixel size and colour profile**, and the GPS if you keep the location. The rest goes, maker notes (which can hold an unblurred preview) included, and the file is checked from its bytes. Padding bytes in a JPEG's last scan stay unless it is compressed again, and the AVIF check was tried on hand-built files and one real file only. A video carries no container metadata.
- **Files it cannot process are refused and listed**, never passed through.

Each promise has a row, with the tests that fail if it breaks, in the audit table `docs/PRIVACY-PATHS.md` in the repository: a record of what was looked for, not a proof that no other path exists.

## What is not promised

- **A miss is silent**: what is not found is neither blurred nor listed ([Accuracy and limits](../accuracy.md)).
- **Only faces and plates are looked for**: cover a house number or a screen with **Add missed area**.
- **A doubtful item that is not placed stays visible** ([How it works](how-it-works.md#doubtful-items)).
- **The comparison video shows the original, unblurred.** Never publish it.
