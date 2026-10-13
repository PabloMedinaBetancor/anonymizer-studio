"""Records the demo video from the real Studio: real server, real engine, the public clips of demo/.

    python3 scripts/make_demo_video.py OUT_DIR            # docs/media/demo.mp4 + demo.jpg are written to OUT_DIR

Chrome (headless, temporary profile) streams its screen with the DevTools screencast (`Page.startScreencast`): every
frame the page really painted, with its timestamp. Nothing is drawn on top: no fake cursor, no invented subtitles. The
clicks and keys are the ones a person would use (tests/ui/drive.py sends real input events). ffmpeg turns each part into a
clip and joins them. The analysis takes about 40 s on the development laptop, so that part is cut to its first seconds and its end (the
cut is a jump in the frame counter of the page, not hidden). Needs websocket-client, ffmpeg, pdftoppm and Chrome.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests" / "ui"))
from real_studio import STAGE, Studio, check_page  # noqa: E402

FPS = 30
SIZE = "1600:1000"


class Recorder:
    """Cuts the screencast into parts and writes each as an mp4 of constant frame rate."""

    def __init__(self, page, work):
        self.page, self.work, self.parts = page, pathlib.Path(work), []

    def start(self):
        self.page.start_screencast(quality=85)

    def stop(self, name, minimum=0.0):
        check_page(self.page)
        end = time.time()
        frames = self.page.stop_screencast()
        if not frames:
            raise RuntimeError(f"no frames for {name}")
        folder = self.work / name
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for index, (stamp, data) in enumerate(frames):
            (folder / f"{index:05d}.jpg").write_bytes(data)
            following = frames[index + 1][0] if index + 1 < len(frames) else end
            lines.append(f"file '{index:05d}.jpg'\nduration {max(following - stamp, 1 / FPS):.4f}")
        lines.append(f"file '{len(frames) - 1:05d}.jpg'")  # concat needs the last file repeated
        (folder / "list.txt").write_text("\n".join(lines) + "\n")
        out = self.work / f"{name}.mp4"
        extra = (
            ["-vf", f"tpad=stop_mode=clone:stop_duration={minimum},scale={SIZE},format=yuv420p"]
            if minimum
            else ["-vf", f"scale={SIZE},format=yuv420p"]
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(folder / "list.txt"),
                *extra,
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                str(out),
            ],
            check=True,
        )
        self.parts.append(out)
        return out


def still_part(image, seconds, out):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-t",
            str(seconds),
            "-i",
            str(image),
            "-vf",
            f"scale={SIZE}:force_original_aspect_ratio=decrease,pad={SIZE}:(ow-iw)/2:(oh-ih)/2:color=0x0f1114,format=yuv420p",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            str(out),
        ],
        check=True,
    )
    return out


def benchmark_png(work):
    """The benchmark shot (80-90 s): docs/media/benchmark.template.html with its {metric} markers filled from bench/results
    by scripts/figures.py (the code behind fill_numbers.py), then captured by Chrome. A marker that cannot be filled, or one
    marked not publishable in the results, stops the video: no figure is ever typed here."""
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import figures as numbers

    template = (root / "docs" / "media" / "benchmark.template.html").read_text(encoding="utf-8")
    text, findings = numbers.fill(
        "benchmark.template.html", template, numbers.load_results(str(root / "bench" / "results"))
    )
    body = re.sub(r"<style>.*?</style>", "", template, flags=re.S)
    findings += numbers.loose_numbers("benchmark.template.html", body, [])
    if findings:
        raise SystemExit("benchmark shot: " + "; ".join(f"{x.rule}: {x.text}" for x in findings))
    page = work / "benchmark.html"
    page.write_text(text, encoding="utf-8")
    png = work / "benchmark.png"
    profile = tempfile.mkdtemp(prefix="anon-bench-chrome-")
    subprocess.run(
        [
            "google-chrome",
            "--headless=new",
            "--no-sandbox",
            f"--user-data-dir={profile}",
            "--hide-scrollbars",
            "--window-size=1600,1000",
            f"--screenshot={png}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    shutil.rmtree(profile, ignore_errors=True)
    return png


def main(out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="anon-demo-"))
    render_dir = STAGE / "anonymized"  # a neutral destination: no home or work folder reaches the video
    shutil.rmtree(render_dir, ignore_errors=True)
    order = {}
    with Studio(lang="en") as studio:
        p = studio.page
        rec = Recorder(p, work)
        studio.import_files(["street-madrid.mp4", "street-taipei.jpg"])

        # 5-15 s: Import (a clip, then a photo, options on the right)
        rec.start()
        p.sleep(1.5)
        p.click(*p.center(".page:not([hidden]) .clip-row:nth-child(2) .clip-card"))
        p.sleep(1.5)
        p.click(*p.center(".page:not([hidden]) .clip-row:nth-child(1) .clip-card"))
        p.sleep(1)
        p.click(*p.center(".page:not([hidden]) .btn-play"))
        p.sleep(2)
        order["import"] = rec.stop("import")

        # 15-25 s: Analyze, live; the middle of the run is cut
        studio.analyze(wait=False)
        rec.start()
        p.sleep(6)
        rec.stop("analyze-live")
        order["analyze"] = rec.parts[-1]
        studio.wait_analyzed()
        rec.start()
        p.sleep(2.5)
        order["analyze-end"] = rec.stop("analyze-end")

        # 25-50 s: Review
        studio.tab("review")
        rec.start()
        p.sleep(1)
        p.click(*p.center(".page-review .timeline-track .bar-item"))
        p.sleep(1.5)
        p.click(*p.center(".review-toolbar .segmented button:nth-child(4)"))  # Boxes
        p.sleep(1)
        p.click(*p.center(".review-toolbar .segmented button:nth-child(3)"))  # Split
        p.sleep(1)
        p.click(*p.center(".page-review .btn-play"))
        p.sleep(1.5)
        p.click(*p.center(".page-review .btn-play"))
        p.sleep(0.5)
        # the photo, which has one Not sure plate: N jumps to it, B blurs it
        p.eval("[...document.querySelectorAll('.clip-card-review')].find(c => /taipei/.test(c.innerText)).click()")
        p.sleep(2)
        p.key("n", "KeyN")
        p.sleep(1.5)
        p.key("b", "KeyB")
        p.sleep(1.5)
        p.click(*p.center(".review-toolbar .segmented button:nth-child(2)"))  # Anonymized
        p.sleep(1.5)
        # back to the video: a face the detector missed, added by hand, then followed along the camera
        p.eval("[...document.querySelectorAll('.clip-card-review')].find(c => /madrid/.test(c.innerText)).click()")
        p.sleep(1.5)
        p.click(*p.center(".review-toolbar .segmented button:nth-child(1)"))  # Original
        p.click(*p.center(".review-toolbar .btn-small:last-child"))
        box = p.eval(
            "(() => { const r = document.querySelector('.page-review .stage').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; })()"
        )
        p.drag(
            box[0] + box[2] * 0.62,
            box[1] + box[3] * 0.55,
            box[0] + box[2] * 0.72,
            box[1] + box[3] * 0.72,
            steps=6,
            pause=0.12,
        )
        p.wait_for("!document.querySelector('.area-form').hidden")
        p.sleep(1.5)
        p.click(*p.center(".area-form .btn-accent"))
        p.sleep(2.5)
        order["review"] = rec.stop("review")
        studio.tab("review")
        p.click(*p.center(".review-toolbar .segmented button:nth-child(3)"))
        p.sleep(2.5)
        studio.shot(work / "split.png")

        # 50-62 s: Export with the comparison video on
        studio.tab("export")
        p.eval(
            f"(() => {{ const f = document.querySelector('#export-destination'); f.value = {json.dumps(str(render_dir))}; f.dispatchEvent(new Event('input')); }})()"
        )
        rec.start()
        p.sleep(1.5)
        p.click(*p.center(".page-export [role=switch]"))
        p.sleep(1.5)
        p.click(*p.center(".page-export .cta .btn-primary"))
        p.wait_for("document.querySelectorAll('.export-files').length >= 1", 600)
        p.sleep(2)
        order["export"] = rec.stop("export")
        errors = studio.js_errors()
        print("page errors:", errors)

    # 0-5 s: the result; 62-80 s: the output files themselves
    parts = [still_part(work / "split.png", 3, work / "intro.mp4")]
    parts += [order[name] for name in ("import", "analyze", "analyze-end", "review", "export")]
    compare = next(render_dir.glob("*.compare.mp4"))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-t",
            "5",
            "-i",
            str(compare),
            "-vf",
            f"scale={SIZE}:force_original_aspect_ratio=decrease,pad={SIZE}:(ow-iw)/2:(oh-ih)/2:color=0x0f1114,format=yuv420p",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            str(work / "compare.mp4"),
        ],
        check=True,
    )
    parts.append(work / "compare.mp4")
    pdf = next(render_dir.glob("street-madrid*.report.pdf"))
    from real_studio import private_hits

    text = subprocess.run(["pdftotext", "-l", "1", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    if private_hits(text):
        raise SystemExit(f"the report shows private strings {private_hits(text)}")
    subprocess.run(
        ["pdftoppm", "-png", "-r", "110", "-f", "1", "-l", "1", "-singlefile", str(pdf), str(work / "report")],
        check=True,
    )
    parts.append(still_part(work / "report.png", 2, work / "report.mp4"))
    parts.append(still_part(benchmark_png(work), 7, work / "benchmark.mp4"))

    (work / "parts.txt").write_text("".join(f"file '{part}'\n" for part in parts))
    video = out_dir / "demo.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(work / "parts.txt"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "27",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(work / "split.png"),
            "-vf",
            "scale=1280:-2",
            "-q:v",
            "4",
            str(out_dir / "demo.jpg"),
        ],
        check=True,
    )
    seconds = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            capture_output=True,
            text=True,
        ).stdout
    )
    print(f"{video}: {video.stat().st_size / 1e6:.1f} MB, {seconds:.0f} s")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/media")
