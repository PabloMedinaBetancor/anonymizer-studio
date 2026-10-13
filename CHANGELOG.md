# Changelog

All notable changes to Anonymizer Studio are in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers are the product's own. [Release notes](docs/release-notes.md) say what each release meant for a user, in longer form.

## [2.2.0] - 2026-10-06

Version 2.2 hides less. Version 2.1 blurred every text region on a vehicle, every doubtful face and the plate-position band, with wide margins, and on outdoor photos with parked cars that covered far more than the plates. The old behaviour is `Options(unsure="blur", fit="loose", plate_band=True)`. What the change costs is on [Accuracy and speed](docs/accuracy.md) and on [Limits](docs/accuracy.md).

### Added

- `Options.unsure` (`"placed"`, `"blur"`, `"keep"`) and `Options.fit` (`"tight"`, `"loose"`).
- A model file, `latin_PP-OCRv5_rec_mobile.onnx` (Apache-2.0), the plate reader. It reads the text regions found on vehicles, so that a plate is told from a word such as "TAXI". It is fetched by `scripts/fetch_models.py` and checked against `models/SHA256SUMS` like every model.
- For custom detectors, a way to say whether a doubtful item is placed like the thing it may be.
- On the Export screen, a note that says how many doubtful items have no decision of yours, in the seven languages.

### Changed

- A doubtful item ("Not sure" in the Studio, the verdict `check` in Python) is blurred by default only when it is placed like a plate or a face. The others are listed in Review, left visible, and counted in the warning `warning.review.check_left_visible`. `Options.unsure="placed"` is the new default.
- A text region on a vehicle is blurred by default only when it scored high, sits where a plate can be, and does not read as a word. Weak regions are united with the regions they touch, and the plate reader brings in a plate that the position rule rejected.
- A sure face that is tiny is blurred only when it is on a person. A doubtful face is blurred when the second look or the first sweep is sure of it and it is not tiny. In video, a doubtful face seen in several sweeps counts as placed.
- The plate-position band is off unless asked for (`Options(plate_band=True)`).
- `Options.fit="tight"` is the default: smaller margins and a narrower, softer edge. The blur inside the box is the same.

### Fixed

- The panel of a plate-shaped text region said "Blurred by default. Please check." next to a "Blur this item" switch that was off. It now says "Blurred only if it is placed like a plate. Please check." in the seven languages.

## [2.1.10] - 2026-10-06

### Fixed

- A video with non-square pixels (a PAL clip shown as 16:9, HDV) came out playing squeezed. The output now keeps the pixel aspect ratio of the source, and so does the comparison video.
- In a photo's list of items, every name from number 10 on lost its number to an ellipsis.
- Review's picture was tiny in a short laptop window.
- A page the Studio refuses threw an error at start.
- The optional native window was taller than a laptop screen. With `pywebview` and GTK but no display, the Studio kept serving to nobody.
- A Studio started with no detection models said nothing until Analyze failed. It now warns when it starts.
- "Remove GPS and location" now says it is for photos.
- A Python error's message was its code. It is now a sentence.
- A photo given to `anonymize_video`, or a video given to `anonymize_image`, was reported as "not a photo or a video". The message now says which it is.
- A destination folder that does not exist was named as a hidden working file, after the whole analysis. It is now refused at once and named.
- `scripts/fetch_models.py` run offline now says what to do.

## [2.1.9] - 2026-10-05

### Fixed

- After `pip install ".[window]"` the Studio opened no window and no browser, and served to nobody. The window now runs on the main thread, closing it ends the Studio, and where it cannot open the page opens in the browser with a line that says why.
- The reports of `x.bmp`, `x.tif` and `x.webp` were all named after the stem, and the later ones got `-2` and `-3`. When two outputs differ only by extension, the report now keeps the media's extension.
- The PDF report's Creator field was the printing browser's user agent. It no longer names your system.
- The first Python example printed an OpenCV warning at every model load.
- A browser that blocks cookies was told to use the address it had used. It is now told to allow cookies for the Studio's address.
- With JavaScript off, the note that the Studio needs it was below the first screen.

### Changed

- The guide says that a video leaves as an MP4 whatever it came in.

## [2.1.8] - 2026-10-05

### Fixed

- A stop during the comparison video left the clip "Done" with no comparison and no word about it. The card now says "Stopped: the comparison video was not made."
- A Studio that had stopped left every button live and showed the browser's own English error. The page now shows a bar, "The Studio is not running", in its language, and dims its screens.
- The refusal "That is not possible in the current state of the file." was replaced by sentences that say what is wrong.
- The PDF report was Letter-sized in every country. It is now A4, and Letter for US English.
- After a stop or a failed clip, the Analyze screen said "done".
- A page the Studio refuses (an address without its session, a bookmark of an earlier start) was in English with a short notice. It is now a page in the language of the browser.
- Review offered "Only items to check" with nothing to check, and a photo's item showed "Visible -", "Length Photo" and two disabled buttons.
- At 320 pixels wide the title lay over the first tab, and Review's legend ran past the window in German.
- After a reload in the middle of an export, the card of the clip being written was headed with the next free name.
- The folder dialog's first place said "Home" in every language.
- A stop that takes seconds left the Stop button as it was. It now says "Stopping...".
- The note on Analyze said each clip opens in Review as soon as it is analysed, and none does. It now says to click the card.
- With JavaScript off the page was blank.
- A file removed from the disk after its analysis left Review with a broken image.
- The PDF's From and To columns wrote a decimal point in every language.
- A Stop in the middle of an export said nothing about how far it got. It now says how many clips were written.

## [2.1.7] - 2026-10-05

### Fixed

- A photo with many items drew its marks on the Analyze screen out of their lane, and an area added in Review was drawn there as a find of the detectors.
- After an export, the clips that had been written said "That is not possible in the current state of the file."
- The card of a written file was headed with the next free name, not the one written.
- A folder import listed capital letters before lower case.
- A file name with a semicolon or a tab shifted the rest of its row in the CSV report when opened in LibreOffice.
- Review told the reader of a photo to click a bar on the timeline and listed keys that do nothing on a photo.
- A file the browser could not read, in a dropped folder, was reported as "The engine did not answer".
- The "Add missed area" form of a photo offered "Follows the camera".
- The cards of the Export list touched each other, and a long file name with no space pushed the headings past the window.
- A folder dropped before the page had its state was uploaded whole, archives and the Studio's own output folder included.
- The PDF and the Review screen said "Certainty 100%" for a plate band. They now show a dash.
- "Done in 0 s" on the card of a photo.
- The Analyze screen of a photo in Careful showed "0%" and "estimating" until the first pass ended.

## [2.1.6] - 2026-10-05

### Added

- The page opens in the language of the browser when the Studio has it, English otherwise. A language chosen once is kept. The PDF report is written in the language of the page.

### Removed

- The Studio no longer offers Maximum thoroughness. It stays in the Python API.

### Fixed

- A folder dropped on Import uploaded every file in it before the server refused it. It is now sorted first.
- The Analyze screen drew a photo's items as bars across the whole width, and "Settings used" named the audio and the gaps for photos.
- The demo's sample list was an empty box, and an empty project's button said "Analyze 0 clips".
- In Firefox, the scrollbar of a side panel was a white strip in the dark page.
- Percentages were written one way in every language.
- The "Add missed area" form squeezed its words and buttons.
- With clips from two folders, Export named the first clip's folder for all of them.
- The example of the output name kept the extension in capitals, and the file did not.
- A destination with an impossible folder name was accepted until the render had started.
- The address that opens the Studio lost the rest of its query.
- The licence of the FFmpeg inside PyAV was stated wrongly in `THIRD-PARTY.md`.
- The benchmark results and scripts named computers by their host names. They now use labels.
- In Review, Boxes view, the picture and the legend contradicted each other.

## [2.1.5] - 2026-10-05

### Changed

- The CSV report is written as UTF-8 with a byte-order mark, so that Excel reads file names with accents.

### Fixed

- The PDF wrote "83 %" in every language, and the page "83%" or "83 %". They now agree.
- A link to a photo, next to the photo, was a second clip of one file.
- The session cookie had no `Secure` flag behind a TLS proxy.
- "Analyze 8 clips" counted the clips that were already analysed with these settings.
- A link to a folder was skipped in silence, with the photos in it.
- Two outputs that differ only in the case of their letters were one file on Windows, macOS and a USB disk.
- On Export, the kind of each written file broke in the middle of a word, and the header was wider than a 320 pixel window.
- A valid photo with more pixels than the safety limit was reported as damaged. It now says it is too large.
- `python3 -m anonymizer.ui --version` was refused as an unrecognised argument.
- A `Range` or `Last-Event-ID` header with thousands of digits, a clip list read while another window imported, and the picture of a clip whose original went away were internal errors.
- A folder of 300 HEIC photos was 300 notices. It is now one.
- Two uploads of one name at the same moment were given one file.

## [2.1.4] - 2026-10-05

### Fixed

- An original that went away during a batch made the whole job fail, and the clip stayed "failed" for ever. It now fails alone.
- A Studio ended with `kill` or by a closed terminal left its work folder, with crops of faces and plates, in the temporary folder. It is now deleted, and the next start removes the folders of Studios that were killed outright.
- Demo mode accepted any number of videos, answered with a 500 that named the server's folder, and showed tracebacks. It now has limits and plain messages.
- The page sent no `Content-Security-Policy`, `Referrer-Policy` or `X-Frame-Options`, and a half-sent request held a thread for ever. Both are closed.
- A render that ran out of room said that the camera and descriptive data could not be removed. It now says the disk is full.
- A named pipe with a photo's name in an imported folder froze the import.
- A photo the person may not read was reported as damaged, and a folder they may not open was skipped in silence.
- The default destination of uploads took a NUL character or ten thousand characters.
- A request for a range of frames that was too large blamed the wrong parameter.
- A job whose engine died said "failed: 0". It now says how the engine died.
- Two quick presses of Play, and resizing the window in Review, raised errors in the page.

## [2.1.3] - 2026-10-05

### Fixed

- Two imports of one folder at the same moment listed every file two or three times.
- Absurd numbers, deep nesting and a NUL character in an address answered with an internal error.
- Adding an area with frames out of range, or a `follow` that does not exist, blamed the wrong field.
- The PDF report was not made when the folder for temporary files had a long path.
- A file name pattern of `.` gave `..jpg`, one with `: " | ? * < >` failed later as "not writable", and the page refused `{index:03d}`.
- After a restart on the same port, the clip list showed the photos of the last run.
- The package, `pip list` and the Studio said version 0.1.0.dev0.
- A page reloaded during an analysis landed on Import with the clip at 0%, and a failed clip lost its reason.
- The Export options and the screen were lost on a reload or a change of language.
- A long file name wrapped in the middle of a word, and a two-character header of the PDF report wrapped in Chinese.
- Review showed "00:00:00:00 / 00:00:00:00" under every photo.
- A file dropped anywhere but the small box under the list made the browser open it and leave the Studio.
- A destination typed without a full path was written in the folder the server was started in. It is now taken from the home folder.
- After the original changed, a browser that had the preview was told "not modified".
- A video whose header overstates its frame count gave errors at its last frames in Review.
- Sizes and counts used English conventions in every language.
- In demo mode a file could only be dragged, and the texts pointed to a button that is not there. The demo also told its visitors the server's folders.
- A second Studio in the same browser logged the first one out.
- The same notice was stacked again and again, and a double click on the cross of a clip took two clips off the list.

## [2.1.2] - 2026-10-04

### Removed

- The Studio no longer offers the Deep level. It stays in the Python API as an experimental value.

### Fixed

- A second window of the Studio did not see what the first one did, and put back in the server the options the first had changed.
- A notice that appeared while a button was held pushed the page from under the pointer.
- Delete and Backspace removed a hand-drawn area at once. They now ask first.
- Toggles, list checkboxes and sliders were smaller than a pointer can hit, and on a tablet or a phone each clip row was as wide as its own text.
- Three slips in the translation of the 2.1.1 texts.
- After Space on a checkbox or a clip card, the keyboard focus was on nothing.
- A settings file that was valid JSON but not an object stopped the Studio at every start.
- The drop zone named videos only, and the English spelled "license" two ways.
- An original that was moved or deleted after the analysis was an internal error. It is now said to be missing.
- The README's first install command failed on a stock Ubuntu.
- The guide described eight of the fourteen columns of the report.
- The tab had no icon, and every load asked the server for one that is not there.
- The example of a file name counted a clip that is not written.

## [2.1.1] - 2026-10-04

### Added

- `scripts/fetch_models.py`. A fresh install had no way to get the models, and the pip of Ubuntu 22.04 could not build the package; the README now says what to do.
- A small cross on each clip of the Import list, to take it off the list. The file is not touched.

### Changed

- A photo too large for the Deep level is refused before anything runs, from the header of the file. It used to be refused after the first pass. Every analysis inside the limits is unchanged.
- Analysing again with other settings asks before it discards a review.

### Fixed

- Import with nothing ticked imported the server's own working directory.
- The pictures of areas from an old analysis outlived it.
- "A detection model could not be loaded" did not say what to do.
- Models fetched into a folder of your own were refused at load.
- Two Portuguese texts were in the wrong variety, and six French apostrophes had the wrong shape.
- The keys B, N and Delete acted on the Review behind a confirmation dialog.
- Clicks on a notice or a clip card were lost during an analysis.
- A dropped folder ended in "The engine did not answer: Failed to fetch", and a stopped server was called an internal error.
- "1 files were not recognized", "1 hidden entries" and "left 1 places unchecked".
- Fast blamed a second look that did not run.
- The refusal of an uploaded file told where the server keeps its files.
- An item of a video could not be selected from the keyboard.
- A long destination ran off the edge of the Export screen on a phone.
- Every option change was an internal error when the settings folder could not be written.
- The Preferences button did nothing.
- Starting the Studio twice ended in a Python traceback.
- A source with a name near 255 bytes made the export end in "The engine stopped unexpectedly", and a NUL character in a name or a negative length crashed the engine.
- Review showed a broken picture and said nothing when the original had changed.
- A double click on Analyze or Render ended with a red "Another job is running".
- An unreadable folder and an old bookmark both said "This request is not allowed".
- The page said nothing when the Studio had stopped.
- Turning off both detectors left a dead Analyze button and a red notice.
- The browser that prints the PDF report contacted its maker's servers. It now runs with its network cut.
- Importing a large folder showed nothing while it worked.
- Hidden controls were shown: "Heavy model only" under Careful, and the doubtful-item buttons on a clip with none.
