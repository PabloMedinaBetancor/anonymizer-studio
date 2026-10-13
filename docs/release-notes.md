# Release notes

These notes say what each release of Anonymizer Studio meant for someone using it. [The changelog](../CHANGELOG.md) lists every change. Read these notes first when you move from one version to another.

## 2.2.0: the Studio hides less

Version 2.1 blurred everything the detectors found on a vehicle, every doubtful face and the place where a plate usually sits on a small vehicle, with wide margins. On outdoor photos of parked cars that covered far more than the plates: logos, wheel hubs, door handles, lettering and the sides of cars all got a blur, and so did plants and poles that a face detector had taken for faces. The person reviewing the result lost sight of the real plates under it.

Version 2.2 changes the defaults so that the plate itself and the face itself are blurred, with a tight margin and a soft edge, and the rest of the picture is left as it was.

- **A doubtful item is blurred only when it is placed like a plate or a face.** The rest are listed in **Review** as **Not sure**, left visible, and counted in a warning at export. [How it works](concepts/how-it-works.md) has the rule.
- **A plate reader tells a plate from a word.** Text on a vehicle is read, so that "TAXI" and a make name are taken out of the default blur, and a plate that the position rule rejected is brought in. The reader needs one more model file, which `scripts/fetch_models.py` downloads.
- **The plate-position band is off.** It put a blur on every small vehicle where no text was found, which is what blurred bare walls. `Options(plate_band=True)` turns it on, for photos.
- **Margins are tight.** The blur reaches a smaller distance beyond the box, and its edge is softer and narrower. The blur inside the box is as strong as before.
- **The Export screen says how many doubtful items have no decision.** Nothing is asked.

The change has a price, and it is on the other side of the same choice. Plates that the text detector scores low or does not find, such as small, far or blurred ones, are left visible by default, and so is some text on a vehicle that is not shaped and placed like a plate. [Accuracy and speed](accuracy.md) says how much each rule hides and blurs by mistake, and [Limits](accuracy.md) says what is missed.

**Going back to 2.1's behaviour.** Use `Options(unsure="blur", fit="loose", plate_band=True)` in Python. The Studio has no switch for it. It offers **Blur all doubtful** on each clip instead.

**Moving from 2.1.** Analyses saved by 2.1 load in 2.2 (the same format, version 3) and keep the decisions made under 2.1's rule: analyse again to get the 2.2 defaults. Files from before 2.1 are refused with `error.analysis.version`. The new model file has to be fetched: run `python3 scripts/fetch_models.py` again. Everything else a script calls is unchanged.

## 2.1.1 to 2.1.10: ten rounds of walking through the Studio

Each of these ten releases came from a walk-through of the previous release, from the README to an export, on the real models, in Chrome and Firefox, in the seven languages and at laptop and phone sizes. What they found was fixed in the next release. The detectors, their thresholds and the second look did not change in this series, and each release was compared with the one before: no analysis result differed. What a user would have met, grouped:

- **Getting started.** A fresh install had no way to get the models, and the pip of Ubuntu 22.04 could not build the package. `scripts/fetch_models.py` was added, and the install steps now say what to do. A Studio started with no models now says so when it starts.
- **When the Studio is not there.** A Studio that had stopped left every button live and showed the browser's English error. The page now says "The Studio is not running", in its language. A second window sees what the first does. A reload keeps the screen, the options and what you typed on **Export**.
- **Privacy of the program itself.** The browser that prints the PDF report contacted its maker's servers, and now runs with its network cut. The PDF report no longer names your system. A Studio ended with `kill` no longer leaves crops of faces and plates in the temporary folder. The page sends security headers, and a half-sent request no longer holds a thread.
- **Files and folders.** Importing was hardened: folders imported twice, links to folders, named pipes, files the person may not read, dropped folders full of archives, hundreds of unsupported files, and names that differ only by case. A file that fails now fails alone, with a sentence that says why.
- **The reports.** The CSV opens correctly in Excel and LibreOffice. The PDF is A4, or Letter for US English, and writes numbers the way the language does. The report of one of several outputs that differ only by extension keeps the media's extension in its name.
- **Output.** A video with non-square pixels (a PAL clip, HDV) keeps its shape. The comparison video has the same shape as the original.
- **The interface.** Language follows the browser. Delete asks before it removes an area. Controls are large enough for a finger, and the clip list fits a tablet and a phone. Errors and warnings are in the language of the page and say what to do.
- **What the Studio no longer offers.** The Deep level, and then the maximum thoroughness, were taken out of the interface because they are too slow and, on video, not repeatable. They stay in the Python API as experimental values.

## 2.1: the interface

Version 2.1 changed only the interface. The results of the engine were compared cell by cell with 2.0, and none differed. It fixed what a client's walk-through of 2.0 found: the word "null" in the file picker, notices that never went away and covered the panels, a damaged file accepted at import and then reported as "Stopped", refusals that did not say why, a layout that needed a wide window, a Studio that stopped itself after one minute, a size estimate far too low, and item numbers that skipped.
