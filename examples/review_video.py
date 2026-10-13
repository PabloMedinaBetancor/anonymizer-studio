# docs-test: real-only - it decides on the doubtful items the real detectors find in the clip
"""Review a video in code: analyse, decide on the doubtful items, add an area, render, save, and render again.

Run it from the repository root:

    python3 examples/review_video.py [CLIP [OUTPUT]]

CLIP defaults to examples/data/street-madrid-5s.mp4 and OUTPUT to reviewed. It writes into OUTPUT: the blurred
video (reviewed.mp4), the saved analysis (review.analysis.json, with its review.analysis.thumbs folder) and a
second video (again.mp4) rendered from the saved analysis without running a detector. It never replaces a file:
delete OUTPUT to run it again. The original is only read.
"""

import os
import sys

import anonymizer

clip = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "examples/data/street-madrid-5s.mp4")
output = sys.argv[2] if len(sys.argv) > 2 else "reviewed"
os.makedirs(output, exist_ok=True)

with anonymizer.Anonymizer() as anon:
    analysis = anon.analyze(clip)
    for item in analysis.items:
        if item.verdict == "check" and not item.blur:
            print(
                f"item {item.item_id}: doubtful and visible, frames {item.first_frame}-{item.last_frame}, {item.reason}"
            )
            analysis.set_blur(item.item_id, True)  # the cautious decision: blur it
    area = analysis.add_area(0, analysis.frames - 1, (755, 210, 860, 242), follow="camera")
    print("added area", area.item_id, area.kind, "blurred:", area.blur)
    result = anon.render(analysis, os.path.join(output, "reviewed.mp4"))
    analysis.save(os.path.join(output, "review.analysis.json"))

print("written:", result.output, "left visible:", result.visible_check)

with anonymizer.Anonymizer() as anon:  # no detector is loaded: the decisions are in the saved file
    again = anon.render(
        anonymizer.load_analysis(os.path.join(output, "review.analysis.json")), os.path.join(output, "again.mp4")
    )
print("written:", again.output)
