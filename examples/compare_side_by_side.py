# docs-test: real-only - it analyses a real clip
"""Anonymise a clip and write a side-by-side video of the original and the result, to check the blur by eye.

Run it from the repository root:

    python3 examples/compare_side_by_side.py [CLIP [OUTPUT]]

CLIP defaults to examples/data/street-madrid-5s.mp4 and OUTPUT to compare. It writes into OUTPUT: the blurred
video (anonymized.mp4) and the comparison video (side-by-side.mp4), whose left half is the original WITHOUT any
blur. Never publish the comparison video. It never replaces a file: delete OUTPUT to run it again.
"""

import os
import sys

import anonymizer

clip = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "examples/data/street-madrid-5s.mp4")
output = sys.argv[2] if len(sys.argv) > 2 else "compare"
os.makedirs(output, exist_ok=True)

blurred = os.path.join(output, "anonymized.mp4")
anonymizer.anonymize_video(clip, blurred, anonymizer.Options(detection="fast"))
side = anonymizer.compare_video(
    clip, blurred, os.path.join(output, "side-by-side.mp4"), captions=("Original", "Anonymized")
)
info = anonymizer.media_info(side)
print("written:", side, f"{info.width}x{info.height}")
