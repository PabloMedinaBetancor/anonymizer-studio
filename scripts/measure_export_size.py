"""How big is an anonymized file next to its original? Prints one JSON line per file with both sizes and their ratio.

    python3 -s scripts/measure_export_size.py demo/street-madrid.mp4 demo/street-taipei.jpg ...

It runs the real engine in fast mode and renders to a temporary folder. The Studio's size estimate on the Export screen is worded
and bounded by what this measures (src/anonymizer/ui/export_size.py); run it again on more files before moving those bounds.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import anonymizer


def main(paths) -> int:
    with tempfile.TemporaryDirectory() as folder, anonymizer.Anonymizer(anonymizer.Options(detection="fast")) as anon:
        for path in paths:
            analysis = anon.analyze(path)
            target = os.path.join(folder, os.path.basename(path))
            anon.render(analysis, target)
            before, after = os.path.getsize(path), os.path.getsize(target)
            info = anonymizer.media_info(path)
            print(
                json.dumps(
                    {
                        "file": os.path.basename(path),
                        "media": info.media,
                        "pixels": info.width * info.height,
                        "frames": info.frames,
                        "bytes_in": before,
                        "bytes_out": after,
                        "ratio": round(after / before, 2),
                        "bits_per_pixel_frame": round(after * 8 / (info.width * info.height * max(1, info.frames)), 4),
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
