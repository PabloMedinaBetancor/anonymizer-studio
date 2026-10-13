"""Screenshots of the four pages in each language, and the layout check of the pseudo-language.

    python3 tests/ui/screens.py shots docs/screens en es     # B1-B4 in each language
    python3 tests/ui/screens.py real docs/screens en es      # the same, from the real engine on the clips of demo/
    python3 tests/ui/screens.py overflow pseudo es en        # reports controls that clip their text

Uses the mock server (a drawn scene, so nothing here is third-party footage), a headless Chrome with a temporary
profile (never the user's).
"""

import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dev_server import serve  # noqa: E402
from drive import Page  # noqa: E402

PORT = 8766
PAGES = ("import", "analyze", "review", "export")
NOT_SURE_BAR = "[...document.querySelectorAll('.page-review .bar-notsure')].pop()"

# Elements whose text is cut off or that stick out of the window.
OVERFLOW = """(() => {
  const found = [];
  for (const el of document.querySelectorAll('.page:not([hidden]) *, .topbar *, dialog[open] *')) {
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    const box = el.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) continue;
    const clipped = el.scrollWidth > el.clientWidth + 1 && !['auto', 'scroll'].includes(style.overflowX) && el.clientWidth > 0 && el.children.length === 0;
    const outside = box.right > innerWidth + 1 || box.left < -1;
    if (clipped || outside) found.push((el.className || el.tagName) + ': ' + (el.textContent || '').trim().slice(0, 50));
  }
  return found.slice(0, 12);
})()"""


def open_page(lang, page, port=9337):
    start = {"import": "", "analyze": "&start=analyze", "review": "&start=review", "export": "&start=export"}[page]
    return Page(f"http://127.0.0.1:{PORT}/?mock=fast&lang={lang}{start}", port=port)


def settle(p, page):
    if page == "import":
        p.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card').length === 3")
    elif page == "analyze":
        p.wait_for(
            "[...document.querySelectorAll('.clip-card-progress .accent-text')].some(e => parseInt(e.textContent) >= 40)"
        )
    elif page == "review":
        p.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9")
        p.eval(
            f"{NOT_SURE_BAR}.scrollIntoView({{block: 'center'}})"
        )  # in a narrow window the timeline is below the fold
        time.sleep(0.3)
        p.click(
            *p.eval(
                f"(() => {{ const r = {NOT_SURE_BAR}.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()"
            )
        )
        p.wait_for("document.querySelector('.item-detail h2')?.textContent.length > 0")
    else:
        p.wait_for("document.querySelectorAll('.export-card').length === 3")
        p.wait_for(
            "document.querySelector('.page-export .btn-primary') && !document.querySelector('.page-export .btn-primary').disabled"
        )
    time.sleep(0.6)


def real_shots(out, langs):
    from real_studio import Studio

    for lang in langs:
        folder = out / lang
        folder.mkdir(parents=True, exist_ok=True)
        with Studio(lang=lang) as studio:
            p = studio.page
            studio.import_files(["street-madrid.mp4", "street-taipei.jpg", "street-westminster.jpg"])
            studio.shot(folder / "B1-import.png")
            studio.analyze(wait=False)
            p.wait_for(
                "[...document.querySelectorAll('.clip-card-progress .accent-text')].some(e => parseInt(e.textContent) >= 45)",
                600,
            )
            studio.shot(folder / "B2-analyze.png")
            studio.wait_analyzed()
            studio.tab("review")
            for clip, name in (("madrid", "B3-review"), ("taipei", "B3-check")):
                p.eval(
                    f"[...document.querySelectorAll('.clip-card-review')].find(c => /{clip}/.test(c.innerText)).click()"
                )
                p.sleep(2)
                if clip == "madrid":  # no item to check in this clip: pick a face bar of the timeline
                    p.click(
                        *p.eval(
                            "(() => { const b = document.querySelectorAll('.page-review .bar-item')[0].getBoundingClientRect(); return [b.x + b.width / 2, b.y + b.height / 2]; })()"
                        )
                    )
                else:
                    p.key("n", "KeyN")
                p.sleep(2)
                studio.shot(folder / f"{name}.png")
            studio.tab("export")
            p.sleep(1.5)
            studio.shot(folder / "B4-export.png")  # the export itself is never started here


def main(argv):
    if argv[1] == "real":
        return real_shots(pathlib.Path(argv[2]), argv[3:])
    threading.Thread(target=serve, args=(PORT,), daemon=True).start()
    mode = argv[1]
    if mode == "shots":
        out = pathlib.Path(argv[2])
        for lang in argv[3:]:
            (out / lang).mkdir(parents=True, exist_ok=True)
            for number, page in enumerate(PAGES, 1):
                p = open_page(lang, page)
                try:
                    settle(p, page)
                    p.screenshot(str(out / lang / f"B{number}-{page}.png"))
                finally:
                    p.close()
    else:
        for lang in argv[2:]:
            for page in PAGES:
                p = open_page(lang, page)
                try:
                    settle(p, page)
                    print(lang, page, p.eval(OVERFLOW))
                finally:
                    p.close()


if __name__ == "__main__":
    main(sys.argv)
