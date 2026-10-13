"""Fetch the detection models of MODELS.md into models/ and check each against models/SHA256SUMS.

    python3 scripts/fetch_models.py                  # every model that is missing or does not match
    python3 scripts/fetch_models.py --models DIR     # into another folder, with a copy of SHA256SUMS (then set ANONYMIZER_MODELS to it)
    python3 scripts/fetch_models.py --only yolox_nano.onnx

The files are not in the repository or in the package (MODELS.md has their licences). A file that is already there and matches its
SHA-256 is left alone; a download that does not match is thrown away and nothing is kept under the model's name. Standard library only;
the only network use is this download, once, and the Studio itself never opens a connection.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from typing import Callable, Dict, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")

# Where each file of models/SHA256SUMS comes from (the column "Source" of MODELS.md; the PP-OCRv6 file is the one RapidOCR 3.9.2 lists).
SOURCES: Dict[str, str] = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "retinaface_r34.onnx": "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx",
    "yolox_nano.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
    "rf-detr-nano-384-coco.onnx": "https://github.com/ankandrew/open-image-models/releases/download/assets/rf-detr-nano-384-coco.onnx",
    "PP-OCRv6_det_small.onnx": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
    "latin_PP-OCRv5_rec_mobile.onnx": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sums(folder: str) -> Dict[str, str]:
    """{file name: SHA-256} from `folder`/SHA256SUMS."""
    found: Dict[str, str] = {}
    with open(os.path.join(folder, "SHA256SUMS"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                found[parts[1].lstrip("*")] = parts[0].lower()
    return found


def download(url: str, target: str, opener: Callable = urllib.request.urlopen) -> None:
    with opener(url, timeout=120) as reply, open(target, "wb") as out:
        while True:
            block = reply.read(1 << 20)
            if not block:
                break
            out.write(block)


def fetch(
    folder: str,
    names: Optional[Sequence[str]] = None,
    sources: Optional[Dict[str, str]] = None,
    sums: Optional[Dict[str, str]] = None,
    opener: Callable = urllib.request.urlopen,
    out=sys.stdout,
) -> int:
    """Make every wanted model present and matching in `folder`. Returns 0 when all are, else 1."""
    sources = SOURCES if sources is None else sources
    sums = expected_sums(folder) if sums is None else sums
    wanted = list(names) if names else sorted(sums)
    failed = 0
    unreachable = 0  # downloads that failed on the way (no network, a server that does not answer)
    os.makedirs(folder, exist_ok=True)
    for name in wanted:
        target = os.path.join(folder, name)
        if name not in sums:
            print(f"{name}: not in SHA256SUMS, so it is not a model of this product", file=out)
            failed += 1
            continue
        if os.path.isfile(target) and sha256_of(target) == sums[name]:
            print(f"{name}: already there and matches", file=out)
            continue
        if name not in sources:
            print(f"{name}: no known source; see MODELS.md", file=out)
            failed += 1
            continue
        partial = target + ".partial"
        try:
            print(f"{name}: downloading from {sources[name]}", file=out, flush=True)
            download(sources[name], partial, opener)
            if sha256_of(partial) != sums[name]:
                print(f"{name}: the download does not match its SHA-256; nothing was kept", file=out)
                failed += 1
                continue
            os.replace(partial, target)
            print(f"{name}: fetched and checked", file=out)
        except OSError as e:
            print(f"{name}: download failed ({type(e).__name__}: {e}); nothing was kept", file=out)
            failed += 1
            unreachable += 1
        finally:
            if os.path.exists(partial):
                os.remove(partial)
    if unreachable:
        print(
            f"{unreachable} of {len(wanted)} models could not be downloaded. Check the internet connection (or a firewall or proxy) "
            "and run this command again: the models already here are kept, and only the missing ones are fetched.",
            file=out,
        )
    return 1 if failed else 0


def ensure_sums(folder: str) -> Dict[str, str]:
    """The checksum list of `folder`: the product checks every model it loads against the SHA256SUMS next to it, so a folder of
    your own gets the list of this checkout (written, not replaced when it is already there)."""
    target = os.path.join(folder, "SHA256SUMS")
    if not os.path.isfile(target):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(MODELS, "SHA256SUMS"), "rb") as reference, open(target, "wb") as out:
            out.write(reference.read())
    return expected_sums(folder)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the detection models into models/ and check their SHA-256.")
    parser.add_argument(
        "--models", default=MODELS, help="folder for the .onnx files (default: models/ of this checkout)"
    )
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these files")
    args = parser.parse_args(argv)
    return fetch(args.models, args.only, sums=ensure_sums(args.models))


if __name__ == "__main__":
    sys.exit(main())
