"""Fetch the v1 plate model (YOLOv9-t 384) into models/, only to reproduce the paired plate comparison (claim C5b) or to run
`Options(plate_path="yolo")`. The product does not ship it and does not need it: its licence cannot be cleared (THIRD-PARTY.md).
The file is git-ignored; the download is checked against the SHA-256 below.

python3 scripts/fetch_reference_plate_model.py [--models FOLDER]
python3 scripts/fetch_reference_plate_model.py --help    # prints this text and does nothing else (no network)"""

import argparse
import hashlib
import os
import sys
import urllib.request
from typing import Optional, Sequence

NAME = "yolo-v9-t-384-license-plates-end2end.onnx"
URL = "https://github.com/ankandrew/open-image-models/releases/download/assets/" + NAME
SHA256 = "888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8"
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the v1 plate model (YOLOv9-t 384) into models/ and check its SHA-256. "
        "Only needed for Options(plate_path='yolo') and the paired plate comparison; the product does not ship it."
    )
    parser.add_argument(
        "--models", default=MODELS, help="folder for the .onnx file (default: models/ of this checkout)"
    )
    args = parser.parse_args(argv)
    models = args.models
    target = os.path.join(models, NAME)
    if os.path.isfile(target) and sha256_of(target) == SHA256:
        print(f"{NAME} is already there and matches.")
        return 0
    os.makedirs(models, exist_ok=True)
    partial = target + ".partial"
    try:
        with urllib.request.urlopen(URL, timeout=120) as reply, open(partial, "wb") as out:
            out.write(reply.read())
    except OSError as err:  # no connection, a firewall, a server that does not answer
        if os.path.exists(partial):
            os.remove(partial)
        print(
            f"{NAME}: the download failed ({err}); nothing was kept. Run it again when the connection is back.",
            file=sys.stderr,
        )
        return 1
    if sha256_of(partial) != SHA256:
        os.remove(partial)
        print(f"{NAME}: the download does not match the expected SHA-256; nothing was kept.", file=sys.stderr)
        return 1
    os.replace(partial, target)
    print(f"{NAME} fetched into {os.path.normpath(models)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
