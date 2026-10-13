"""The recorded walk-through for the demo video: one still per step, with its caption and how long it stays on screen.

    python3 tests/ui/walkthrough.py OUT_DIR [lang]

Writes OUT_DIR/NN-name.png and OUT_DIR/steps.json (the shot list of the demo video).
ffmpeg turns it into the video: see "Assembling the walk-through" in that file. Uses the mock server, so the
footage is a drawn scene and nothing third-party.
"""

import json
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dev_server import serve  # noqa: E402
from drive import Page  # noqa: E402

PORT = 8767
STEPS = []


def shot(p, out, name, caption, seconds):
    index = len(STEPS) + 1
    path = f"{index:02d}-{name}.png"
    time.sleep(0.5)
    p.screenshot(str(out / path))
    STEPS.append({"file": path, "caption": caption, "seconds": seconds})


def main(out, lang="en"):
    out.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=serve, args=(PORT,), daemon=True).start()
    p = Page(f"http://127.0.0.1:{PORT}/?mock=1&lang={lang}", port=9338)
    try:
        p.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card').length === 3", 60)
        shot(p, out, "import", "Import: three clips, Faces and License plates, Standard, follow camera motion", 10)
        p.click(*p.center(".page:not([hidden]) .cta .btn-primary"))
        p.wait_for(
            "[...document.querySelectorAll('.clip-card-progress .accent-text')].some(e => parseInt(e.textContent) >= 40)",
            120,
        )
        shot(p, out, "analyze", "Analyze: live boxes, a PLATE? not sure, time left, found so far", 10)
        p.wait_for(
            "document.querySelectorAll('.page:not([hidden]) .clip-card-progress').length === 3 && [...document.querySelectorAll('.clip-card-progress')].every(c => c.querySelector('.accent-text') === null && !c.innerText.match(/Queued|Cola/))",
            240,
        )
        p.eval(
            "[...document.querySelectorAll('.tabs button')].find(b => b.disabled === false && b.getAttribute('aria-current') !== 'page' && /Review|Revisar/.test(b.textContent)).click()"
        )
        p.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9", 60)
        shot(p, out, "review", "Review: the timeline shows every face and plate; striped bars are not sure", 6)
        p.key("n", "KeyN")
        shot(p, out, "review-not-sure", "N jumps to the next item to check", 6)
        p.key("b", "KeyB")
        shot(p, out, "review-blur", "B blurs it: one item left to check", 6)
        p.click(*p.center(".review-toolbar .segmented button:nth-child(3)"))
        shot(p, out, "review-split", "Split view: original and anonymized side by side", 6)
        p.click(*p.center(".review-toolbar .segmented button:nth-child(2)"))
        p.click(*p.center(".review-toolbar .btn-small:last-child"))
        r = p.eval(
            "(() => { const r = document.querySelector('.page-review .stage').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; })()"
        )
        p.drag(r[0] + r[2] * 0.1, r[1] + r[3] * 0.1, r[0] + r[2] * 0.3, r[1] + r[3] * 0.3)
        p.wait_for("!document.querySelector('.area-form').hidden")
        shot(
            p,
            out,
            "review-add-area",
            "Add missed area: drag a box, choose how long and whether it follows the camera",
            8,
        )
        p.click(*p.center(".area-form .btn-accent, .area-form .btn-primary"))
        time.sleep(0.5)
        p.eval(
            "[...document.querySelectorAll('.tabs button')].find(b => /Export|Exportar/.test(b.textContent)).click()"
        )
        p.wait_for("document.querySelectorAll('.export-card').length === 3", 60)
        p.click(*p.center(".page-export .toggle"))
        shot(p, out, "export", "Export: reports, comparison video on, file names, sizes", 8)
        p.click(*p.center(".page-export .cta .btn-primary"))
        p.wait_for("document.querySelectorAll('.export-files').length === 3", 240)
        shot(p, out, "export-done", "Rendered: the video, the reports and the comparison, each its own file", 8)
    finally:
        p.close()
    (out / "steps.json").write_text(json.dumps(STEPS, indent=2) + "\n")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), *(sys.argv[2:3]))
