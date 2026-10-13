# Studio UI protocol (local server <-> page)

Version 1, 03/10/2026. Covers the events and commands of the engine and the public API. The server is `src/anonymizer/ui/server.py` (standard library only, imports only the top-level `anonymizer`
package). The page is `src/anonymizer/ui/static/`. **This file is the contract between them: a change to it must state
the old and the new shape.**

## 0. Conventions

- Base URL `http://127.0.0.1:<port>/`. The port is free and chosen at start; the server prints and opens the URL.
- Every JSON body, request or response, is UTF-8. Every JSON message the server originates carries `"v": 1`.
- The server sends **no texts**, only i18n keys with `params`; the page translates (section 2).
- Frame numbers are 0-based. Times are seconds (float). Boxes are `[x0, y0, x1, y1]` in pixels of the **displayed**
  source frame (after rotation), floats. A photo is a clip with `frames = 1`; its items have `first_frame = last_frame
  = 0`.
- `clip_id` is chosen by the server (`"c1"`, `"c2"`, ... per session). Items are `item_id` integers, unique inside a
  clip. An id never changes and is not reused in the session.
- Everything under `/api/` needs the session token (section 1). Only `/`, `/static/*` and `/i18n/*` do not.

## 1. Session, security, errors

- On start the server makes a random token (32 hex characters) and opens `http://127.0.0.1:<port>/?t=<token>`. Serving
  `/` with the right `t` sets the cookie `anon_session_<port>=<token>; HttpOnly; SameSite=Strict; Path=/` and answers
  `302 /` (the token leaves the address bar). Every `/api/` request, including media tags (`<video>`, `<img>`) and the
  event stream, is authorised by that cookie. The page never reads or sends the token itself.
- A request whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>`, or a `POST`/`PUT`/`PATCH`/`DELETE` whose `Origin`
  header is present and is not the server's own origin, is refused (`403`, `error.request.forbidden`).
- Errors: an HTTP status and a body

```json
{"type": "error", "v": 1, "code": "error.output.exists", "params": {"path": "/home/user/anonymized/x.mp4"}, "detail": "technical text for the log"}
```

| Status | When | Typical codes |
|---|---|---|
| 400 | malformed JSON, missing or wrong field | `error.request.invalid` (`params.field`) |
| 403 | no or wrong cookie, bad Host/Origin, demo mode forbids it | `error.request.forbidden`, `error.browse.denied`, `error.demo.disabled` |
| 404 | unknown clip, item, file or route | `error.not_found` |
| 409 | a job is already running, or the clip is in the wrong state, or an analysis would discard decisions | `error.job.busy`, `error.clip.state`, `error.analysis.discards` |
| 413 | upload over the limit | `error.upload.too_large` |
| 415 | unsupported or unreadable file (listed, never skipped) | `error.media.heic`, `error.media.raw`, `error.media.gif`, `error.media.format`, `error.media.empty`, `error.media.too_large` (more pixels than Pillow's limit, about 179 MP: `megapixels` and `limit` in the params), `error.media.denied` (the file exists and the person may not read it), `error.media.unreadable_photo`, `error.media.unreadable_video`, `error.media.unsupported` |
| 422 | the request is well formed but refused | `error.analysis.incomplete`, `error.source.changed`, `error.output.exists`, `error.demo.limit` (`params.limit` is `photos`, `clips`, `seconds` or `height`), `error.output.not_writable`, `error.output.disk_full` (a render that found no room: `params.path` is the folder; also what the engine's "metadata not removed" becomes when the folder has less free space than the original) |
| 500 | bug in the server | `error.internal` |

`detail` is for the log, never shown as it is. Error codes are the `code` of `anonymizer.AnonymizerError` where one
exists, so the page sees the same keys as a Python user.

## 2. Static files and i18n

- `GET /` -> `static/index.html`. `GET /static/<path>` -> the files under `src/anonymizer/ui/static/`, with their media
  type, `Cache-Control: no-cache`. No directory listing; no `..`.
- `GET /i18n/<lang>.json` -> `{"<key>": "<text with {placeholders}>", ...}`, the file `src/anonymizer/ui/i18n/<lang>.json`
  (flat, one level). The same file holds the page keys and the engine keys (`error.*`, `warning.*`, `video.warning.*`,
  `photo.warning.*`, `review.warning.*`, `plate.reason.*`, `face.reason.*`, `manual.reason.*`). Unknown `<lang>` -> 404;
  the page falls back to `en`. Plural forms: keys `name.one` / `name.other` (the page picks with `Intl.PluralRules`).
  Placeholders are `{name}`; the page replaces them from `params` and leaves a missing one visible as `{name}`.
- The list of languages the server has is in `GET /api/state` (`languages`). `en` always exists.
- `formats` in `GET /api/state` is what the door takes: `photo` and `video` (lists of extensions, lower case, with the dot), `unsupported` (`{extension: reason key}` for the formats known not to be supported: HEIC, RAW, GIF and others) and `output_folder` (the folder the Studio writes into, which a folder import does not walk). A page that is given a folder by a drop sorts it with this before it sends a byte: photos and videos go up, an unsupported format is refused with its reason, any other file is left out and counted (`warning.import.ignored`), as the server's own folder import does.

## 3. State and settings

### `GET /api/state`

The whole picture: the page calls it on load and after a reconnect (`state.resync`, section 6).

```json
{
  "v": 1,
  "version": "2.1.8",
  "mode": "local",
  "languages": ["en", "es"],
  "capabilities": {
    "pdf_report": {"available": true, "reason": null, "params": {}},
    "comparison_video": {"available": true},
    "browse": true,
    "upload": true
  },
  "demo_limits": null,
  "slowdown": {"careful": null},
  "idle_stop": {"seconds": 600, "seconds_with_clips": 21600},
  "settings": { "...": "see below" },
  "clips": [ "ClipSummary, section 4" ],
  "job": null,
  "seq": 41
}
```

- `mode`: `"local"` or `"demo"`. In demo mode `browse` is false, `demo_limits` is `{"max_photos": 10, "max_clips": 10, "max_seconds": 20,
  "max_height": 720, "max_upload_mb": 100}` (`max_clips`: files in the session, videos included), and the settings are not saved (section 3). Otherwise `demo_limits` is null.
  The demo is somebody else's computer: its answers name files and not the server's folders (`plan` and `render` give file names in
  `output` and `outputs`, `destination` is empty, `analysis.json` names its source by file name) and `default_destination` is empty.
- `slowdown.careful` is how many times slower the Careful tier ran than `fast` on a clip analysed both ways in this session, rounded to one
  decimal; `null` until that has happened. It is measured on the user's machine, never a constant; the Detection control prints a factor
  only when it is not `null`. (There is no `slowdown.maximum` since 2.1.6: the Studio does not offer Maximum thoroughness.)
- `idle_stop` is how long the server waits with no page open before it stops by itself, with nothing loaded and while clips are loaded (section 9); the Import screen says so.
- The Studio does not offer Deep (2.1.2): there is no `capabilities.deep`, no `slowdown.deep`, no `heavy_only` setting, and a `detection` of `deep` is 400 `error.request.invalid`. It stays an experimental value of the Python API.
- `capabilities.pdf_report.available` is false when no Chromium-based browser is found; `reason` is then an i18n key
  (`report.pdf.no_browser`). The page disables the PDF chip and shows the reason as its tooltip. The PDF is printed from an
  HTML page by the Chrome, Chromium or Edge already on the computer (headless, throw-away profile); the page is built in the
  language of the settings from the same i18n texts (`report.*` and the reason keys).
- `job` is null, or `{"job_id": "j3", "kind": "analyze"|"render", "state": "running"|"paused"|"stopping",
  "clips": ["c1","c2"], "current": "c1"}`.
- `seq` is the id of the last event sent (section 6).

### `GET /api/settings`, `PUT /api/settings`

```json
{
  "language": null,
  "faces": true,
  "plates": true,
  "thoroughness": "standard",
  "detection": "careful",
  "gap_fill": "camera",
  "blur": "gaussian",
  "strength": 1.0,
  "extra_margin": 0.0,
  "remove_gps": true,
  "default_destination": "/home/user/Anonymizer",
  "threads": 0
}
```

`PUT` takes any subset, validates it as `anonymizer.Options` does (400 `error.request.invalid` with `params.field`) and
answers with the full object. Saved to `settings.json` in the user folder (`$XDG_CONFIG_HOME/anonymizer-studio/` or
`~/.config/anonymizer-studio/`; `%APPDATA%\AnonymizerStudio\` on Windows), written atomically. Demo mode keeps the settings
in memory only. The values are: `thoroughness` `standard` (`maximum` is a value of the Python API; the Studio does not offer it since 2.1.6, and 400 `error.request.invalid` refuses it); `detection` `fast` or `careful`; `gap_fill` `camera`, `straight` or `hold`; `blur` `gaussian`,
`pixelate` or `solid`; `faces`, `plates` and `remove_gps` booleans; `language` one of `state.languages`, or `null` until the person has chosen one (the page then follows the browser's language: the first of `navigator.languages` the Studio has, English if none; a language chosen once, English included, is kept). `strength` is 1.0 to 2.0 (the safe minimum is 1.0; the page's slider cannot go below it) and
`extra_margin` 0 to 0.5. `threads` 0 = automatic. `default_destination` is where uploaded files are written (section 5)
and the fallback when a clip has no source folder.

The `blur`, `strength` and `extra_margin` settings are also what `preview_frame` shows for the Anonymized and Split views.

## 4. Clips

```json
ClipSummary = {
  "clip_id": "c1",
  "name": "street.mp4",
  "media": "video",
  "origin": "path",
  "source_dir": "/home/user/Videos",
  "duration_s": 12.4,
  "frames": 310,
  "width": 1920,
  "height": 1080,
  "fps": 25.0,
  "codec": "h264",
  "has_audio": true,
  "has_gps": false,
  "size_bytes": 41234567,
  "thumb_url": "/api/clips/c1/thumb?v=1791154210189789",
  "source_url": "/api/clips/c1/source",
  "status": "imported",
  "analysis": null
}
```

- `origin`: `"path"` (the server reads the file where it is; `source_dir` set) or `"upload"` (copied into the session
  folder; `source_dir` null).
- `photo`: `media = "photo"`, `frames = 1`, `duration_s = 0`, `fps = 0`, `codec` = image format (`jpeg`, `png`...).
- `status`: `imported` -> `queued` -> `analyzing` -> `analyzed` -> `rendering` -> `rendered`, or `failed`. `analyzed` is the Review state. A `rendered` clip can still be reviewed;
  changing a decision sets it back to `analyzed`. A stopped job puts its clips back where they were (`imported` or `analyzed`).
- `analysis` is null until analyzed, then `{"items": 7, "to_check": 2, "unchecked_frames": 0, "rev": 5, "settings": {"faces": true, "plates": true, "thoroughness": "standard", "detection": "careful", "gap_fill": "camera"}}` (section 7); `settings` is what the analysis was made with (null when the file has changed since), so that the page counts as "to analyze" only the clips that Analyze would do again.
  **`unchecked_frames > 0` means the clip cannot be exported as clean**: the page shows the warning, and the server
  refuses its render (422 `error.analysis.incomplete`).
- While a clip is `analyzing` its summary also carries `progress` (the last `clip.progress` of its analysis without `clip_id` and
  `phase`: `fraction`, `frame`, `eta_s`, and `live`, `found`, `stage`, `zones` when they were sent) and `items_so_far` (the `ItemSummary`
  of each `clip.item` sent so far), and a `failed` clip carries `failure` (`{"code": ..., "params": ...}`, what its `job.error` said):
  a window that is opened or reloaded in the middle of a job shows the clip as it is, not at 0 % and not "Failed" with no reason.
  A clip in an export carries `render`: `{"phase": "queued" | "render" | "compare", "fraction": 0.4, "output": "x_anonymized-2.mp4"}` (`output`: the name of the file being written, never a path; a window opened in the middle of the export names it) while its file is being written, and
  `{"phase": "done", "outputs": [...], "report": {...}, "comparison": ..., "comparison_stopped": false, "seconds": 3.2}` (what its `clip.rendered` said) once it is; it is
  absent before any export and after the clip is analysed again. A window reloaded in the middle of an export shows the progress and the
  files already written, as the other window does. A clip that waits for its turn in an export is `queued` with its `analysis` and `render`.
  `thumb_url` ends in `?v=` and the number of this server's start: a Studio started again on the same port gives `c1` to another
  file, and the browser keeps the picture of a clip for an hour.

### `GET /api/browse?path=<abs path>` (local mode only)

The browser cannot give disk paths, so the server lists folders and the Import dialog draws them. No `path`: the
roots (home, desktop, downloads, videos, pictures where they exist, and the drives on Windows).

```json
{
  "path": "/home/user/Videos",
  "parent": "/home/user",
  "roots": [{"name": "Home", "path": "/home/user"}],
  "entries": [
    {"name": "trips", "kind": "dir", "items": 12},
    {"name": "street.mp4", "kind": "video", "size_bytes": 41234567, "mtime": 1790000000},
    {"name": "plate.jpg", "kind": "photo", "size_bytes": 2412345, "mtime": 1790000000},
    {"name": "IMG_0001.HEIC", "kind": "unsupported", "size_bytes": 3100000, "mtime": 1790000000}
  ],
  "truncated": false
}
```

Folders first, then files, each sorted by name (case-insensitive). Hidden entries (starting with `.`) are not listed.
`kind` is decided from the extension with the lists of the public API (`anonymizer.VIDEO_EXTENSIONS`,
`PHOTO_EXTENSIONS`, `UNSUPPORTED_EXTENSIONS`); `unsupported` (HEIC, RAW, GIF...) is shown greyed with its reason, an i18n key in
the entry's `reason` (`error.media.heic`, `error.media.raw`, `error.media.gif`, `error.media.format`; `params` are `name` and `extension`).
Other files are not listed.
`items` (the count of entries in a folder) is omitted when the folder is not readable. At most 2,000 entries;
`truncated` says so. `404 error.not_found` when the path is not a folder; an unreadable folder is `403 error.browse.denied` with
`params.path`. Demo mode: 403
`error.demo.disabled`.

### `POST /api/clips` (import)

```json
{"paths": ["/home/user/Videos/street.mp4", "/home/user/Videos/trips"], "recursive": true}
```

Files and folders. A folder adds the supported files directly inside it (and below when `recursive`, default true). A path already
imported is not imported twice: its existing clip is returned. Answer `200`:

```json
{"clips": ["ClipSummary", "..."], "rejected": [{"path": "/home/user/Videos/IMG_0001.HEIC", "code": "error.media.unsupported", "params": {"name": "IMG_0001.HEIC"}, "detail": "..."}]}
```

Unsupported or unreadable files are **listed in `rejected`**, never silently skipped (E14), each with the reason in its `code` (`ui/intake.py`): the format (HEIC, RAW, GIF, other), an empty file, or a photo or video that cannot be read (damaged, incomplete, several images). The check opens the file with the engine's own readers, so a file that is imported is one the analysis can open; a file that makes the probe fail with anything else is listed too, and does not cost the others. When a folder holds files that are neither photos nor videos nor a known unsupported format, one more entry says so: `{"path": "", "code": "warning.import.ignored", "params": {"count": 3, "extensions": ".txt, .vob"}, "detail": "<the first names>"}` (hidden files are left out of that count). Entries that start with a dot (files, and folders with all they hold) are not imported, and one more entry says how many: `{"path": "", "code": "warning.import.hidden", "params": {"count": 2}, "detail": ""}`. Importing is synchronous
(it probes the files; it does not analyze) and refused with 409 while a job is running. In demo mode: 403
`error.demo.disabled`; use `samples` or upload.

### `POST /api/clips` with samples (demo mode, also allowed locally if `demo/` exists)

`{"samples": ["street-1"]}` imports the public clips listed by `GET /api/samples` -> `[{"sample_id", "name", "media",
"duration_s", "licence", "author", "source_url"}]` (from `demo/SOURCES.csv`; the page shows the licence).

### `PUT /api/upload?name=<file name>`

Body: the raw bytes of the file (`Content-Type: application/octet-stream`, `Content-Length` required, streamed to disk,
never held in memory). This is how drag-and-drop works: the browser gives file contents, not paths. Answer: `{"clips": [ClipSummary],
"rejected": []}` as above, `origin = "upload"`. One file per request. Limits: demo mode, `demo_limits`; local mode, free
disk space (413 `error.upload.too_large` with `params.max_mb` when it would not fit). Demo limits are refused with 422
`error.demo.limit` (`params.limit`: `"photos" | "seconds" | "height" | "size"`) and the copy is deleted.

### `DELETE /api/clips/<clip_id>`

Removes the clip from the session (and its analysis and thumbnails; never the source file). 409 while a job uses it.
Answer `{"ok": true}`.

### `GET /api/clips`

`{"clips": [ClipSummary...]}` (same as in `state`).

## 5. Source video, frames and thumbnails

All of these are plain `GET`s that the browser can put in `src=`.

- `GET /api/clips/<id>/source` -> the original file, `Accept-Ranges: bytes`. `Range: bytes=a-b`, `a-` and `-n` are
  answered `206 Partial Content` with `Content-Range`; a range that cannot be satisfied is `416`. `Content-Type` by extension
  (`video/mp4`, `video/quicktime`, `video/x-matroska`, `video/webm`, `image/jpeg`...). The server opens the file read-only and
  never writes to it. The page tests `video.canPlayType` with `codec`; a clip the browser cannot decode (HEVC, many
  AVI/MKV) is shown frame by frame with `preview` instead of playing.
- `GET /api/clips/<id>/thumb` -> JPEG of the clip, 240 px wide, taken at 10 % of the duration (the first frame for
  photos). Made on first request and cached.
- `GET /api/clips/<id>/preview?frame=<n>&view=<view>&w=<px>&rev=<n>` -> `image/jpeg` of one frame. `view`:
  - `original`: the frame as it is;
  - `anonymized`: the frame with the blur of the current decisions and the `blur` / `strength` / `extra_margin` settings;
  - `boxes`: the frame with each item's box drawn (solid for "blur", dashed for "check", colour by kind) and no blur;
  - `split`: the left half original, the right half anonymized, divided by a line at the middle.
  `w` is the output width in pixels (default 960, 160 to the source width; height keeps the aspect ratio). `rev` is the
  clip's analysis version counter; it starts at the server's start time (microseconds) and goes up by one per change, so a server restarted on the same port never repeats a `rev` the browser may have cached. The answer carries `Cache-Control: private, no-cache` and an `ETag` made of `rev`, the settings that shape the blur (style, strength, extra margin, gap filling, classes), the frame, the view and the width; the browser asks again with `If-None-Match` and gets `304` when nothing changed, so a change of setting is never shown with the old blur.
  `anonymized`, `boxes` and `split` need the clip analyzed (409 `error.clip.state` otherwise). The blur is drawn with the
  same engine function and boxes as a render, on the decoded frame (a render also goes through H.264, so the pixels are
  not bit-identical). With variable frame rate the frame is the one nearest to the number.
  `frame` out of range -> 400 `error.request.invalid`, `params.field = "frame"`.
- `GET /api/clips/<id>/boxes?from=<n>&to=<m>&rev=<n>` -> the final boxes of every item that is blurred (decision on),
  per frame, for drawing the blur over the playing `<video>` (an approximation of the blur, exact when paused with
  `preview`):

```json
{"rev": 5, "frame_from": 0, "frame_to": 49, "fps": 25.0, "frames": {"0": [{"item_id": 3, "kind": "face", "box": [412.0, 88.5, 470.2, 160.0], "shape": "ellipse"}], "1": []}}
```

  At most 600 frames per call (400 `error.request.invalid` beyond). Boxes already include the margins and the extra margin.
  `shape` is `"ellipse"` (faces) or `"rect"`. Frames with nothing to blur have an empty list.
- `GET /api/clips/<id>/items/<item_id>/thumb?rev=<n>` -> JPEG crop of the item (at most 160 px); for an area added by hand, the frame it starts at cropped to its box with a little context (at most 240 px), 404 once it is removed. `thumb_url` carries the `rev` of the analysis it belongs to: with it the answer is `private, max-age=3600`, without it `no-cache`; a new analysis has a new address.

## 6. Events (server -> page)

### `GET /api/events` (Server-Sent Events)

`Content-Type: text/event-stream`. The page opens it once with `new EventSource("/api/events")` and keeps it open. Each
event is

```
id: 42
data: {"type": "clip.progress", "v": 1, "clip_id": "c1", "phase": "analyze", "fraction": 0.31, "frame": 96, "frames": 310, "eta_s": 38}

```

(one `data:` line of JSON, blank line after). `id` grows by one per event. Open the stream first, then load `GET /api/state`, and ignore events whose `id` is not
greater than the state's `seq`. The browser sends `Last-Event-ID` on
reconnect and the server replays what came after it from a ring of the last 5,000 events. If the id is older than the ring, or the
server restarted, the first event is `{"type": "state.resync", "v": 1}` and the page reloads `GET /api/state`. A line
`: ping` is sent every 15 s so a dead connection is noticed. A reconnect is silent for the user unless it fails for
more than 10 s, then the page says the engine stopped answering.

### The events

These are the events of the engine; `clip_id` is the server's id, never a file name.

| `type` | Fields | Notes |
|---|---|---|
| `job.started` | `job_id`, `kind` (`"analyze"`/`"render"`), `clips`: `[{clip_id, name, media, duration_s, frames, width, height, fps, has_audio, has_gps}]` | first event of a job |
| `clip.started` | `clip_id`, `phase` | a clip of the job begins (`analyze` or `render`); status changes to `analyzing`/`rendering` |
| `clip.progress` | `clip_id`, `phase` (`analyze`, `render` or `compare`), `fraction` 0..1, `frame`, `frames`, `eta_s` (null if unknown), `live?`, `found?`, `stage?` (`first` or `second`; the engine's `deep` stage is not used by the Studio), `zones?` (`{"done", "total"}` in stage `second`) | monotonic per clip and phase; at most 4 per second; `live` and `found` below |
| `clip.item` | `clip_id`, `item`: ItemSummary | one per item, after the analysis ends, in order of first frame |
| `clip.analyzed` | `clip_id`, `items`, `to_check`, `unchecked_frames`, `rev`, `settings`, `frames`, `analysis_url` | the clip is ready for Review; `frames` is the real count of the analysis (the count read from a Matroska, WebM or AVI header can be one over; the clip summary then carries the real one) |
| `clip.rendered` | `clip_id`, `outputs`: `[path]`, `report`: `{csv?, json?, pdf?}`, `comparison?`: path, `comparison_stopped?`: true when the person stopped the export while the comparison video was being made, `seconds` | paths on the server's disk; in demo mode they are download URLs (section 8) |
| `clips.added` | `clips`: `[ClipSummary]` | clips newly added to the session (import, upload, samples); not sent for a file that was already there. Every open page gets it, the one that asked included, so a page applies a clip it already has once |
| `clip.removed` | `clip_id` | the clip left the session |
| `item.changed` | `clip_id`, `item`: ItemSummary, `rev`, `to_check` | a decision on one item (the answer of `PATCH .../items/<id>`, sent to every page) |
| `items.changed` | `clip_id`, `items`: `[ItemSummary]`, `rev`, `to_check` | the bulk decision on the doubtful items |
| `area.added` | `clip_id`, `item`: ItemSummary, `rev` | an area drawn in Review |
| `area.removed` | `clip_id`, `item_id`, `rev`, `to_check` | an area taken out |
| `settings.changed` | `settings`: all the settings, `by`: the `X-Studio-Client` header of the `PUT /api/settings` that changed them (empty when it had none) | a change of settings, sent only when a value changed; a window that sends `X-Studio-Client` ignores the events that carry its own id. A window sends only the settings it changed (`PUT` merges), never all of them |
| `job.warning` | `code`, `clip_id?`, `params` | the work went on; shown to the user |
| `job.error` | `code`, `clip_id?`, `params`, `detail` | a clip or the job failed; `detail` goes to the log only. With a `clip_id` of a clip being analyzed, the page shows that clip "Failed" with the reason (not "Stopped") |
| `job.finished` | `job_id`, `status` (`"done"`/`"stopped"`/`"failed"`), `summary`: `{clips, analyzed, rendered, failed, to_check, seconds}` | last event of a job |
| `job.state` | `state` (`"running"`/`"paused"`/`"stopping"`) | after `pause`, `resume`, `stop` is acted on |
| `state.resync` | - | reload `/api/state` |

`live` and `found` (in `clip.progress`, phase `analyze`, video only, about every 15 frames) show what is in view while the
analysis runs. **Both are provisional**: the items of the finished analysis (`clip.item`) are the result, and the numbers here can
differ from them (the engine merges and drops tracks at the end). `live` is what is in view at `frame`, so the page can
draw it over `preview?view=original&frame=<frame>`:
`[{"kind": "face"|"plate", "box": [..], "score": 0.97, "state": "found"|"unsure"}]` (`found`: the class rule already
confirms it; `unsure`: not yet, drawn dashed as "PLATE? 41"). `found` counts, by kind, the things seen at least twice so far:
`{"face": 3, "plate": 1}`. Both are absent from the last progress event (fraction 1.0), from photos and from `render`; the
page must work without them.

ItemSummary:

```json
{
  "item_id": 3,
  "kind": "plate",
  "judgement": {"verdict": "check", "reason": "plate.reason.unconfirmed", "params": {}},
  "first_frame": 12,
  "last_frame": 88,
  "first_s": 0.48,
  "last_s": 3.52,
  "frames": 77,
  "max_score": 0.41,
  "found_by": "light",
  "second_score": null,
  "decided_by": "engine",
  "blur": true,
  "reviewed": false,
  "manual": false,
  "thumb_url": "/api/clips/c1/items/3/thumb"
}
```

- `kind`: `"face"`, `"plate"` or `"manual"`. `judgement.verdict`: `"blur"` or `"check"` (items the engine drops are never listed).
  `reason` is an i18n key. `max_score` 0..1; the page shows it as a percentage ("Certainty 41 %"). `found_by` is `light`, `second`,
  `deep` or `manual`; the page badges `second` as "Found by the second look" and shows `second_score` (0..1, or null). `decided_by`
  is `engine` until a person sets the item (`PATCH`, the bulk decision or an added area), then `person`.
- `blur` is the **current decision** (what a render does): true for every item the engine is sure of ("blur") and, since 2.2, for a "check" item only when its class module says it is placed
  (`judgement.params.placed`; with `Options.unsure="blur"`, 2.1's rule, for every "check" item), until the user changes it; true for manual areas.
- `reviewed` becomes true once the user has set the item explicitly (`PATCH` below). `to_check` counts items with verdict
  `check` and `reviewed = false`. A clip is "Ready" when `to_check = 0` and `unchecked_frames = 0`.
- For photos `first_s = last_s = 0`.

Event order for one analyzed clip: `job.started`, `clip.started`, `clip.progress`... (fraction increasing, last one 1.0),
`clip.item`..., `clip.analyzed`, then the next clip, then `job.finished`. Rendering: `job.started`, `clip.started`,
`clip.progress`..., `job.warning`..., `clip.rendered`, ..., `job.finished`. After `stop`, `job.finished` has status
`stopped` and no output exists under a final name.

## 7. Job commands and Review commands

One job at a time. A command while another job runs is `409 error.job.busy`.

### `POST /api/job/analyze`

```json
{"clips": ["c1", "c2"], "options": {"faces": true, "plates": true, "thoroughness": "standard", "detection": "careful", "gap_fill": "camera"}}
```

`options` is optional (the settings are used) and any missing field takes the setting's value. `discard` (a boolean, default `false`)
says that the person agreed to lose what they decided: when other options would replace the analysis of a clip that has decisions or added
areas (`reviewed` items), the answer is `409 error.analysis.discards` with `{"count": clips, "decisions": n}` and nothing starts, until the
same request comes with `"discard": true`. Answer `202`
`{"job_id": "j1", "clips": ["c1","c2"], "estimate_s": 372}`; events follow. A clip already analyzed with the same options
and the same source fingerprint is not analyzed again: its `clip.analyzed` event is sent at once (E8). A `detection` of `deep`, or a `heavy_only`, is 400 `error.request.invalid` (the Studio does not offer Deep). `estimate_s` is the
time the page shows as "About X on this PC": frames times the per-frame time measured on this machine in earlier clips of
the session, or `null` until one has been measured (the page then shows nothing, not a guess).

### `POST /api/job/pause`, `/api/job/resume`, `/api/job/stop`

No body. Answer `{"ok": true, "state": "paused"|"running"|"stopping"}`; `job.state` follows as an event. Stop acts within 2 s between frames
(E8); nothing under a final name is left behind. 409 `error.clip.state` when no job runs.

### `GET /api/clips/<id>/analysis`

```json
{"clip_id": "c1", "rev": 5, "frames": 310, "fps": 25.0, "width": 1920, "height": 1080, "unchecked_frames": 0, "to_check": 2, "items": ["ItemSummary", "..."]}
```

Items in order of first frame. 409 `error.clip.state` before the analysis. `GET /api/clips/<id>/analysis.json` downloads the
saved analysis file (`Analysis.save`).

### `PATCH /api/clips/<id>/items/<item_id>` (set_item)

`{"blur": true}`. Answer: the new ItemSummary plus `"rev"` and `"to_check"`: `{"item": {...}, "rev": 6, "to_check": 1}`. `rev`
grows by one on every change of decisions; previews with an older `rev` are stale. 404 for an unknown item.

### `POST /api/clips/<id>/doubtful` (set_doubtful)

`{"blur": false}` keeps every "check" item of the clip visible, `{"blur": true}` blurs them all; the page asks the person to
confirm first. Items with verdict `blur` and manual areas are not touched. Every "check" item becomes `reviewed`. Answer:
`{"items": [ItemSummary, ...], "rev": 7, "to_check": 0}` (the items that were set). `blur` must be a JSON boolean: anything
else, or another key, is 400 `error.request.invalid`.

### `POST /api/clips/<id>/areas` (add_area)

```json
{"frame_from": 40, "frame_to": 120, "box": [300, 200, 380, 250], "follow": "camera"}
```

`box` is at `frame_from`, in pixels of the source, `follow` is `"camera"` or `"fixed"`; following the camera stops at a change of shot (the area stays where it was and follows the new shot from there). For photos `frame_from = frame_to = 0`. Only the part of the box inside the picture is kept. Answer `201`:
`{"item": ItemSummary (kind "manual", verdict "blur", blur true, reviewed true), "boxes": {"40": [..box..], "41": [..]}, "rev": 6}`
where `boxes` is the item's final box per frame (as in section 5). 400 `error.request.invalid` for an empty box, a box wholly outside the picture or with a number that is not finite, a bad `follow` or
frames out of range; the `field` of the error says which value: `frame_from`, `frame_to`, `follow` or `box`.

### `DELETE /api/clips/<id>/areas/<item_id>` (remove_area)

Answer `{"ok": true, "rev": 7, "to_check": 1}`. 404 if the item is not a manual area (detected items are turned off with
`PATCH`).

Decisions are kept by the server in memory and saved to the clip's analysis file after each change (debounced, 1 s).

### `GET /api/clips/<id>/preview` is the engine's `preview_frame`

The command `preview_frame` of the spec is the `GET` of section 5; its answer is the image, not a URL.

## 8. Render

### `POST /api/render/plan` (no side effects)

What the Export dialog shows live: the file names, the free names, the sizes and what is blocked.

```json
{"clips": ["c1", "c2"], "options": {"destination": "/home/user/Videos/anonymized", "name_pattern": "{name}_anonymized", "reports": {"pdf": true, "csv": true, "json": true}, "comparison": false}}
```

Answer:

```json
{
  "destination": "/home/user/Videos/anonymized",
  "folders": ["/home/user/Videos/anonymized"],
  "unwritable": [],
  "writable": true,
  "free_bytes": 52000000000,
  "clips": [
    {"clip_id": "c1", "output": "/home/user/Videos/anonymized/street_anonymized.mp4", "exists": false, "estimate_bytes": 39000000, "estimate_low_bytes": 9000000, "blocked": null, "to_check": 0},
    {"clip_id": "c2", "output": "/home/user/Videos/anonymized/plate_anonymized-2.jpg", "exists": true, "estimate_bytes": 2400000, "estimate_low_bytes": 1000000, "blocked": null, "to_check": 1}
  ],
  "total_bytes": 41400000,
  "total_low_bytes": 10000000
}
```

The size is a range (`ui/export_size.py`: the output is written again, so its size follows the picture, not the original file): `estimate_low_bytes` to `estimate_bytes`, and the free space is checked against the top.
`blocked` is null or `{"code": "error.analysis.incomplete", "params": {"frames": 12}}` (frames that could not be analyzed) or
`{"code": "error.source.changed", "params": {}}` or `{"code": "error.output.name_too_long", "params": {"limit": 235}}` (the output name, in bytes, would not fit a
file system once the render's hidden working name is added). `output` is already the name the render will use: an existing file gets `-2`, `-3`, ...
(nothing is overwritten); `exists` says a suffix was added. `name_pattern` has the placeholders `{name}` (source name without extension) and
`{index}` (also with a number format such as `{index:03d}`); nothing else is allowed (`{name.__class__}` is a 400), and it must not contain a path separator, a character that Windows cannot save in a file name (`< > : " | ? *`, outside the fields) or be only dots and spaces (all 400 with the field `name_pattern`). The options may also carry `blur`, `strength`, `extra_margin`, `remove_gps`, `gap_fill` and `language` (one of `state.languages`: the page's own language, which the PDF is written in; omitted, the language of the settings, or English if none was chosen) and `paper` (`a4`, the default, or `letter`: the size of the PDF report's page; the page sends `letter` when the browser's language is `en-US`, `a4` for every other): they are used for this render in place of the saved settings, so an export started before the page saved a change still uses what is on the screen. `destination` omitted: `<source_dir>/anonymized`, or
`default_destination` for uploads. Output extension: `.mp4` for video (H.264), the source's for photos. The
estimate is the source bitrate times the duration for video, the source size for photos; it is labelled "about" by the page.

### `POST /api/render`

Every body is checked for keys: a key the server does not read (at the top level of `analyze`, `plan`, `render`, `PATCH` of an item, `POST` of `doubtful`, `POST` of an area and `POST /api/clips`, or inside the `options` of the first three) is 400 `error.request.invalid` with that key as `params.field` (`options.<key>`), so a page that sends a field in the wrong place fails at once instead of being ignored. `tests/test_page_server_contract.py` runs the page's own `api.js` and `export-logic.js` to build the bodies and sends them to the real server.

The same body as `plan`, plus the output options of the settings (`blur`, `strength`, `extra_margin`, `remove_gps` are taken from the
settings and may be overridden in `options`). Answer `202` `{"job_id": "j2", "clips": ["c1","c2"], "outputs": {"c1": "/..."}}`.
The request is refused whole with 422 when any clip is `blocked` in the plan (the page removes those first); the
destination is created when missing, and an unwritable one is 422 `error.output.not_writable` (`params.path`).

**Items marked Not sure start blurred only when they are placed (2.2; 2.1: always).** The render asks for no acknowledgement: `acknowledge_visible` and `error.review.pending` no longer exist, and a body that carries `acknowledge_visible` is refused like any unknown key (400 `error.request.invalid`). `to_check` in the plan (the same number as in the analysis) counts the Not sure items nobody has decided on; the page says how many doubtful items will be exported as they started and you can still review them. With `Options.unsure="keep"`, or `"placed"` and an item that is not placed, a `job.warning` `warning.review.check_left_visible` (`params.count`) says how many Not sure items are visible in that output, as the Python API does (`Result.warnings`, and a `JobWarning`).

Per clip the server renders to a hidden partial file, checks it, renames it, writes the reports next to it
(`<name>_anonymized.report.csv|json|pdf`), then the comparison video when asked (`<name>_anonymized.compare.mp4`). Each is
its own file, never overwritten. `clip.rendered` lists them. A failed PDF (no browser) does not fail the clip: a `job.warning`
`report.pdf.unavailable` is sent and `report.pdf` is absent. A failed render writes nothing under a final name.

The comparison video (original on the left, anonymized on the right, a caption over each, the length and audio of the
original) is made after the render, from the source and the file just written. While it is made, `clip.progress` events
have `phase: "compare"` (after the `render` ones, fraction 0 to 1 again). Captions are always English ("Original",
"Anonymized"): they are burned into the picture with a built-in font. A photo gets no comparison and a `job.warning`
`comparison.photo_skipped`. If making it fails, a `job.error` is sent for the clip, the render stays as it is and
`clip.rendered` has no `comparison`; a stop during it keeps the render too, and says so: `comparison_stopped: true` in `clip.rendered` (and in the clip's `render`) and a `job.warning` `warning.comparison.stopped` (`params.name`). **The comparison video shows the original without blur**: when it is written a `job.warning` `warning.comparison.contains_original` (`params.file`) is sent, the page says so on the option, and the JSON and PDF reports of that clip carry the note `report.note.comparison_video` (JSON: `"notes": [{"code": ..., "params": {"file": ...}}]`; the CSV is rows only). The reports of a clip are written after its comparison video for that reason.

Demo mode: outputs go to the temporary session folder; the `outputs`, `report` and `comparison` of `clip.rendered` are URLs of the
form `/api/downloads/<file_id>/<name>` (`GET`, `Content-Disposition: attachment`). The folder is deleted when the session ends or
after 30 minutes without requests, and the server keeps no logs.

### `GET /api/clips/<id>/report?format=json|csv`

The privacy report of the current decisions, without rendering (the page can show counts before export). JSON: `{"version": 1,
"items": [ReportRow...]}`; CSV with its header. 409 before the analysis.

## 9. Lifetime, watchdog, shutdown

- `python -m anonymizer.ui [--demo] [--port N] [--no-browser] [--engine process|thread]`. The server stops when the page has been gone for 10 minutes (6 hours while clips
  are loaded, since their analyses and decisions live only in memory) with no job running (no event stream connected and no request), or on
  `POST /api/quit`. In a job it never stops by itself. It prints the two waits when it starts, and the state carries them (`idle_stop`).
- By default each job runs its
  engine in a worker process (`python -m anonymizer.ui.worker`, one per job, JSON lines on its pipes; the models are
  loaded once per job). A watchdog looks every 30 s: a worker that has sent nothing for 10 minutes (any event counts,
  and the time of a Pause does not) is killed by its pid. The job then ends with `job.error` `error.engine.silent`
  (`params.minutes`), the clip it was on is `failed` (analysis) or stays `analyzed` (render), and `job.finished` has
  status `failed`. A worker that dies (a native library crash, the process killed) gives `job.error`
  `error.engine.crashed` with its exit code and the end of its error output in `detail`. The server, the other clips and
  their analyses survive both, and the next job starts a new worker. A Stop that the worker does not obey in 30 s is a
  kill too (the job ends `stopped`). `--engine thread` keeps the engine in the server process (no watchdog is possible
  there; used by tests and for debugging).
- `POST /api/quit` -> `{"ok": true}` and the process ends after the answer.

## 10. What the API gives the server (changes to the public API, made in PR-2)

The server uses only `anonymizer`. These were added to `__all__` deliberately (A1 pin updated, API.md
regenerated), as part of PR-2; the page does not depend on how:

1. `anonymizer.media_info(path) -> MediaInfo` (duration, frames, size, fps, codec, audio, GPS, size in bytes) and
   `anonymizer.thumbnail_jpeg(path, width) -> bytes (JPEG)`.
2. `Analysis.boxes(frame_from, frame_to) -> dict` (the final boxes of section 5) and `Analysis.preview(view, frame, width,
   options) -> bytes (JPEG)`.
3. `Item.params` (reason parameters) and `Analysis.thumbnail_jpeg(item_id) -> bytes`.
4. `Progress.live` and `Progress.found` (what is in view and how many things were found so far), optional.
5. `anonymizer.frame_jpeg(path, frame, width)` (a frame as it is), `Analysis.report(clip)` (the report before rendering)
   and the extension lists `VIDEO_EXTENSIONS`, `PHOTO_EXTENSIONS`, `UNSUPPORTED_EXTENSIONS`.
6. The `Warning` event of the API is now `JobWarning` (03/10/2026), so it no longer hides the Python builtin; its JSON
   `type` stays `job.warning`.

7. `anonymizer.compare_video(...)` (the comparison video) and, in `Analysis.save`, the item thumbnails (JPEG files in a
   folder next to the JSON, read back by `load_analysis`: the worker process hands its analysis over this way).

All of them exist (03/10/2026). Nothing here changes the JSON above.

## 11. Example session

```
GET  /                                  -> 200 index.html (cookie set)
GET  /api/state                         -> {"mode":"local", "clips":[], "job":null, ...}
GET  /api/events                        -> (stream stays open)
GET  /api/browse?path=/home/user/Videos    -> {entries:[...]}
POST /api/clips {"paths":["/home/user/Videos/street.mp4"]}  -> {"clips":[{"clip_id":"c1",...}],"rejected":[]}
POST /api/job/analyze {"clips":["c1"]}  -> 202 {"job_id":"j1","clips":["c1"],"estimate_s":null}
   event job.started, clip.started, clip.progress x N, clip.item x 7, clip.analyzed {"items":7,"to_check":2,"unchecked_frames":0,"rev":0}, job.finished
GET  /api/clips/c1/analysis             -> {"items":[...]}
PATCH /api/clips/c1/items/3 {"blur":true} -> {"item":{...,"blur":true,"reviewed":true},"rev":1,"to_check":1}
GET  /api/clips/c1/preview?frame=40&view=split&rev=1   -> image/jpeg
POST /api/render/plan {"clips":["c1"],"options":{}}    -> {"clips":[{"output":".../street_anonymized.mp4","blocked":null}],...}
POST /api/render {"clips":["c1"],"options":{"reports":{"csv":true,"json":true,"pdf":false}}} -> 202
   event job.started, clip.started, clip.progress x N, clip.rendered, job.finished
```
