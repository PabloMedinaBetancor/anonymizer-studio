"""The PDF privacy report: an HTML page printed to PDF by the Chromium-based browser already on the computer."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence

import anonymizer

from .engine import no_window

BROWSER_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge")
WINDOWS_BROWSERS = (
    r"Google\Chrome\Application\chrome.exe",
    r"Microsoft\Edge\Application\msedge.exe",
    r"Chromium\Application\chrome.exe",
)
MAC_BROWSERS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
TIMEOUT_SECONDS = 90
# The browser that prints the report is started with its network cut: left alone, a fresh profile of Chrome opens connections to Google's
# servers (DNS lookups and TLS on 443 and 80 for its own services), which the "nothing leaves this computer" promise does not allow. Every
# request goes to a proxy that nothing listens on and no name is looked up; what is left is a UDP "is there an IPv6 route" probe that sends nothing.
OFFLINE_FLAGS = (
    "--proxy-server=socks5://127.0.0.1:1",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-default-apps",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--no-pings",
    "--disable-breakpad",
)
BAND_NOTE = "report.note.band"  # under the table, when a plate band is in it
POSITION_BAND = "plate.reason.position_band"  # the reason of a plate band; the engine gives such an item the constant 1.0 as its score
KIND_KEYS = {"face": "report.kind.face", "plate": "report.kind.plate", "manual": "report.kind.manual"}
STYLE = (
    "body{font:14px/1.45 system-ui,Segoe UI,Helvetica,Arial,sans-serif;color:#1a1d24;margin:32px}"
    "h1{font-size:22px;margin:0 0 4px}p{margin:2px 0 14px;color:#4a5160}"
    "table{border-collapse:collapse;width:100%;font-size:12.5px}"
    "th,td{border-bottom:1px solid #d5d9e0;padding:6px 8px;text-align:left;vertical-align:top}"
    "th{background:#f0f2f5;white-space:nowrap}td.n,th.n{text-align:right;white-space:nowrap}tr.visible td{color:#7a4a00}"
    "p.small{font-size:12px;margin:14px 0 0}"
)


PAPERS = {"a4": "A4", "letter": "Letter"}  # the sizes of the page of the report (CSS `@page size`)


def find_browser() -> Optional[str]:
    """A Chromium-based browser that can print a page to PDF, or None."""
    for name in BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    candidates = list(MAC_BROWSERS)
    if sys.platform == "win32":
        roots = [os.environ.get(v) for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
        candidates += [os.path.join(root, rel) for root in roots if root for rel in WINDOWS_BROWSERS]
    return next((c for c in candidates if os.path.isfile(c)), None)


DECIMAL_COMMA = (
    "es",
    "de",
    "fr",
    "it",
    "pt",
)  # the languages of the Studio that write a decimal comma (the page does too: "Done in 3,5 s")


def decimal_for(language: str) -> str:
    return "," if language in DECIMAL_COMMA else "."


def clock(seconds: float, decimal: str = ".") -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}".replace(".", decimal)


def text_of(texts: Dict[str, str], key: str, **params: object) -> str:
    template = texts.get(key, key)
    for name, value in params.items():
        template = template.replace("{" + name + "}", str(value))
    return template


def report_html(
    clip: str,
    rows: Sequence["anonymizer.ReportRow"],
    texts: Dict[str, str],
    video: bool,
    day: date,
    comparison_file: Optional[str] = None,
    paper: str = "a4",
    decimal: str = ".",
) -> str:
    """A standalone page (all text escaped) with the summary and one row per item, in the language of `texts`, for the paper `paper` ("a4",
    or "letter" for a reader whose browser is in US English: Chromium prints Letter when the page does not say, to every country)."""

    def t(key: str, **params: object) -> str:
        return html.escape(text_of(texts, key, **params))

    blurred = sum(1 for r in rows if r.blurred)
    head = [t("report.col.item"), t("report.col.kind"), t("report.col.decision"), t("report.col.reason")]
    if video:
        head += [t("report.col.from"), t("report.col.to")]
    head += [t("report.col.frames"), t("report.col.certainty")]
    numeric = {t("report.col.from"), t("report.col.to"), t("report.col.frames"), t("report.col.certainty")}
    lines = []
    for number, r in enumerate(
        rows, 1
    ):  # 1 to n in time order, as the Studio numbers them; the engine's ids skip dropped candidates
        decision = "report.decision.blurred" if r.blurred else "report.decision.visible"
        reason = t(r.reason)
        if r.kind != "manual" and r.decided_by == "person" and r.verdict != "drop":
            # The Reason column is the detectors' judgement and the Decision column what was done; when the person overruled it the two
            # read as a contradiction ("Blurred by default. Please check." next to "Left visible"), so the page says who decided
            reason += " " + t("report.reason.confirmed" if r.blurred else "report.reason.turned_off")
        cells = [str(number), t(KIND_KEYS.get(r.kind, "report.kind.manual")), t(decision), reason]
        if video:
            cells += [clock(r.first_s, decimal), clock(r.last_s, decimal)]
        band = r.reason == POSITION_BAND
        cells += [
            str(r.frames),
            "\u2013" if band else t("review.cert.sure", percent=round(r.max_score * 100)),
        ]  # the page's own way of writing it; a plate band (a guess from where a plate usually sits) has no certainty: its 1.0 is a constant
        tags = [("td", " class=n" if h in numeric else "") for h in head]
        row = "".join(f"<{tag}{cls}>{cell}</{tag}>" for (tag, cls), cell in zip(tags, cells, strict=True))
        lines.append(f"<tr{'' if r.blurred else ' class=visible'}>{row}</tr>")
    header = "".join(f"<th{' class=n' if h in numeric else ''}>{h}</th>" for h in head)
    nothing = t("report.none")
    body = "".join(lines) or f"<tr><td colspan={len(head)}>{nothing}</td></tr>"
    summary = t("report.summary", items=len(rows), blurred=blurred, visible=len(rows) - blurred)
    title = t("report.title")
    generated = t("report.generated", date=day.isoformat())
    note = f"<p><b>{t('report.note.comparison_video', file=comparison_file)}</b></p>" if comparison_file else ""
    band_note = f"<p class=small>{t(BAND_NOTE)}</p>" if any(r.reason == POSITION_BAND for r in rows) else ""
    size = PAPERS.get(paper, "A4")
    return (
        f"<!doctype html><html><head><meta charset=utf-8><title>{title}</title><style>@page{{size:{size}}}{STYLE}</style></head><body>"
        f"<h1>{title}</h1><p>{html.escape(clip)}</p><p>{generated}</p>"
        f"<p>{summary}</p>{note}<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>{band_note}</body></html>"
    )


SHORT_TEMP_ABOVE = 60  # bytes: a longer folder for temporary files and Chromium cannot make the socket it starts with


def browser_environment() -> Dict[str, str]:
    """The environment the printing browser runs in. Chromium makes `<temp>/com.google.Chrome.XXXXXX/SingletonSocket` at start, and a
    socket path cannot pass about 104 bytes: with a long folder for temporary files (a sandbox, a deep TMPDIR) it aborted ("Socket
    path too long"), the report was not made and its crash dump stayed in the person's browser data. Its temporary files go to a
    short folder when the usual one is long."""
    env = dict(os.environ)
    if (
        os.name == "posix" and len(os.fsencode(env.get("TMPDIR") or "/tmp")) > SHORT_TEMP_ABOVE
    ):  # where Chromium puts it
        for short in ("/tmp", "/var/tmp"):
            if os.path.isdir(short) and os.access(short, os.W_OK):
                env["TMPDIR"] = short
                break
    return env


def html_to_pdf(browser: str, html_path: str, pdf_path: str) -> None:
    """Print the page to `pdf_path` with a throw-away browser profile. Raises RuntimeError when no PDF came out."""
    with tempfile.TemporaryDirectory(prefix="anonymizer-pdf-") as profile:
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            # the PDF's Creator field is the browser's user agent: the person's operating system and browser version, in a file that is sent on
            "--user-agent=Anonymizer Studio",
            *OFFLINE_FLAGS,
        ]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            command.append("--no-sandbox")  # Chromium refuses to start as root otherwise
        command.append(Path(html_path).as_uri())
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
                env=browser_environment(),
                **no_window(),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"the browser took more than {TIMEOUT_SECONDS} s") from e
    if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) < 100:
        raise RuntimeError(f"the browser wrote no PDF (exit {done.returncode}): {done.stderr[-300:]!r}")
    with open(pdf_path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise RuntimeError("the browser's output is not a PDF")
