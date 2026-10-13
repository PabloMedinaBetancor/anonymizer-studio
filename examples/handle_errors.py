# docs-test: real-only - it compares the class and the code of each error with the real models
"""Turn the errors of the API into an exit code, and show each one on a file made to fail.

Run it from the repository root:

    python3 examples/handle_errors.py

It writes into a folder named errors-demo (a fresh one is made, so a second run needs the old one deleted). It makes
the files that fail itself: a text file called .heic, a photo cut short, a photo that changes after its analysis.
The exit code is 0 when every case gave the class and the code the page shows, and 1 when one differs, so it can be
used as a smoke test of an install.
"""

import os
import shutil
import sys

import anonymizer

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


def exit_code(path, target):
    """0 when the photo is written, 1 for a file that is not a usable photo, 2 for a problem of the install, 3 for the rest."""
    try:
        anonymizer.anonymize_image(path, target, options)
    except anonymizer.UnsupportedMedia:
        return 1
    except (anonymizer.ModelLoadError, ValueError):
        return 2
    except (anonymizer.AnonymizerError, OSError):
        return 3
    return 0


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
attempt(
    "model folder missing",
    lambda: anonymizer.Anonymizer(options, models_dir=os.path.join(folder, "none")).analyze(
        os.path.abspath(os.path.join(folder, "cut.jpg"))
    ),
)
attempt("bad option", lambda: anonymizer.Options(strength=0.5))

for label, (name, code) in outcomes.items():
    print(f"{label:22s} {name} {code or ''}".rstrip())
print("exit code for scan.heic:", exit_code(os.path.join(folder, "scan.heic"), os.path.join(folder, "o.jpg")))

EXPECTED = {
    "same name again": ("OutputExists", "error.output.exists"),
    "folder missing": ("FileNotFoundError", None),
    "not a photo": ("UnsupportedMedia", "error.media.unsupported"),
    "cut short": ("UnsupportedMedia", "error.media.unsupported"),
    "source missing": ("AnonymizerError", "error.source.missing"),
    "source changed": ("SourceChanged", "error.source.changed"),
    "model folder missing": ("ModelLoadError", "error.model.load"),
    "bad option": ("ValueError", None),
}
differences = [label for label, got in outcomes.items() if got != EXPECTED[label]]
for label in differences:
    print("differs:", label, outcomes[label], "instead of", EXPECTED[label])
sys.exit(1 if differences else 0)
