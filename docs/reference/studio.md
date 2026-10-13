# The Studio

Every screen, control, key and setting of the Studio, in the interface's own words. To learn it by doing, follow [Your first video](../tutorials/first-video.md).

The Studio is a page served by a program on your computer, with four screens: **Import**, **Analyze**, **Review** and **Export**, in the tabs at the top.

## Starting and stopping

`anonymizer-studio`, or `python3 -m anonymizer.ui`, prints an address and opens it in your browser ([Command line](command-line.md) has the options). Open the printed address: it carries a session, and a bookmark of an earlier start is refused. The Studio stops by itself after ten minutes without a page open, or six hours while clips are loaded, and at once on Ctrl+C or a closed terminal. Anything not exported is then gone.

## Import

**Import** browses this computer for photos, videos or whole folders, or takes them dropped on the screen; a dropped file is uploaded as a copy, so its output goes to your default destination. Every file is a "clip" here, a photo too. HEIC, RAW, GIF and other formats it cannot process are refused at once, with the reason.

- **Faces**, **License plates**: what to look for. At least one must be on.
- **Detection**: **Fast**, "The light models only, as in version 1.", or **Careful**, "Recommended. Also looks again, with a heavier model, at the places the light models doubted." After a clip has been analysed both ways on this computer, **Careful** shows "N× slower".
- **Fill gaps between detections**: **Follow camera motion**, **Straight line** or **Hold last box**.
- **Style**: **Gaussian blur**, **Pixelate** or **Solid box**. **Strength**: ×1.0 to ×2.0; 1.0 is the floor. **Extra margin**: +0% to +50% of the box on each side.
- **Remove GPS and location from photos**: on by default. Other metadata is dropped anyway.

The settings are kept between sessions. **Analyze N clips** starts the analysis. Changing a detection setting after an analysis asks "Analyze again and discard your review?", because a new analysis replaces the decisions and the added areas.

## Analyze

Each clip card shows its progress, then "Analyzed" with its item count. **Pause**, **Resume** and **Stop** act on the work; a photo in Careful says "Stopping..." until its first pass ends. A dashed "FACE?" or "PLATE?" box marks an item the detectors are not sure about. A banner at the end says if the second look left places unchecked; a clip with frames that could not be checked cannot be exported as clean. Click an analysed card to open it in **Review**.

## Review

**View** switches between **Original**, **Anonymized**, **Split** and **Boxes** (every item outlined, dashed when left visible). Items are numbered from 1 in time order, as in the PDF report. A video has a timeline with one bar per item, striped for a **Not sure** item; a photo has a list.

- **Next item to check** (`N`; `Shift+N` goes back) selects the next **Not sure** item that has no decision.
- **Blur all doubtful** and **Keep all doubtful visible** decide every **Not sure** item of the clip, after a confirmation. **Only items to check** and **Only second-look finds** filter the list.
- **Add missed area**: drag a box over what the detectors missed, then choose how long it lasts and whether it **Follows the camera** or **Stays in place**. It is blurred until you turn it off or remove it (`Delete` or `Backspace`, after asking; `Escape` stops drawing).

The panel on the right shows the selected item: a crop, its reason in plain words, its **Certainty** (a dash for a plate band), the **Second look score** if the heavy model looked at it, and the switch **Blur this item** (`B`). `Space` plays or pauses, `.` and `,` select the next and previous item of any kind, and `Left` and `Right` move one frame, ten with `Shift`.

## Export

**Export** draws the blur from the analysis, without running the detectors again; each doubtful item with no decision of yours goes out as marked.

- **Save to**: by default a folder named `anonymized` next to each original, created when missing.
- **File name**: a pattern, `{name}_anonymized` by default; `{index}` and `{index:03d}` number the files. A name that exists gets `-2`: "Nothing is ever overwritten."
- **Privacy reports**: **PDF**, **CSV** and **JSON**, next to the output. **Comparison video**: the original, unblurred, beside the result: "do not publish it".

## Settings, languages and demo mode

The settings are kept in `settings.json`, in `$XDG_CONFIG_HOME/anonymizer-studio/` or else `~/.config/anonymizer-studio/`: the **Import** controls, `language`, `default_destination` (for uploaded files, `~/Anonymizer`) and `threads`. The Studio speaks the seven interface languages: English, Spanish, French, German, Italian, Portuguese and Simplified Chinese. It follows your browser until you choose one with the language button, and the PDF report is in the language of the page. `--demo` starts it for shared use, with nothing saved; one program is one session.
