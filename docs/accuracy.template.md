# Accuracy and limits

How often Anonymizer Studio hides the faces and plates it is meant to blur, what each figure was measured on, and what the product does not do. Read it before you decide whether the Studio fits your footage. The figures are read from the results files by a script; none is typed here.

## The figures

| What was measured | Result | Condition |
|---|---|---|
| Faces in photos hidden by default, Careful | {results.faces_v2.independent_parts.careful.pct}% of {results.faces_v2.independent_parts.careful.faces} (Wilson interval {results.faces_v2.independent_parts.careful.wilson95}) | 2.1's settings, on photos never used to choose a number: the 30 blind photos of the next row and the sealed half of WIDER FACE |
| Faces in 30 blind photos of the kind a real-estate agent takes, Careful | {results.faces_v2.parts.blind.careful.pct}% of {results.faces_v2.parts.blind.careful.faces} (Wilson interval {results.faces_v2.parts.blind.careful.wilson95}) | 2.1's settings. The figure to quote for listing photos. There is no blind figure for the 2.2 defaults |
| Faces in photos, the 2.2 defaults | {configs.v2.2.faces.recall_pct}% of {configs.v2.2.faces.recall.n} faces in {configs.v2.2.faces.photos} photos | The development part, which the rules were fitted on and whose labels started from the detector's own proposals: in-sample and optimistic |
| Readable plates in photos, the 2.2 defaults | {configs.v2.2.plates.readable_recall_pct}% of {configs.v2.2.plates.readable_recall.n} plates in {configs.v2.2.plates.photos} photos | The development part: in-sample |
| Plates of the OpenALPR EU set, the 2.2 defaults | {configs.v2.2.eu.readable_recall.hits} of {configs.v2.2.eu.readable_recall.n} hidden | Never used to choose a number; the boxes are OpenALPR's own |
| Readable plates of 60 street photos, the 2.2 defaults | {configs.v2.2.streets.readable_recall.hits} of {configs.v2.2.streets.readable_recall.n} hidden | Never used to choose a number. Too few plates for a rate, so none is quoted |
| Faces in video covered at least 80 percent, Careful | {results.video_v2.cells.careful.hits.covered_80} of {results.video_v2.cells.careful.faces} | Three sealed clips, one of them a crowd that holds most of the faces. In video the second look did not measurably add to Fast ({results.video_v2.careful_over_fast.gain_points_80} points, not significant). Plates in video were not measured |
| Photos without people on which something was hidden, Careful | {results.faces_v2.cells.careful.false_hidden_photos_without_people.per_100_photos} per 100 ({results.faces_v2.cells.careful.false_hidden_photos_without_people.photos_affected} of {results.faces_v2.cells.careful.false_hidden_photos_without_people.photos} photos); Fast {results.faces_v2.cells.fast.false_hidden_photos_without_people.per_100_photos} | 2.1's settings. This rate is not claimed to be low; the 2.2 defaults trade it for some plates and faces left visible |
| Seconds per photo for face detection alone, median | Careful {results.faces_v2.cells.careful.seconds_per_photo_median}, Fast {results.faces_v2.cells.fast.seconds_per_photo_median} | The {results.faces_v2.speed.photos} sealed face photos, one run on the development laptop (an Intel Core i7-8550U), which was not idle: read them as indicative. With plates a photo takes seconds, and a video much more. The Studio shows an estimate while it analyses, and Careful costs more than Fast |

In the rows with 2.1's settings, `Options(unsure="blur", fit="loose", plate_band=True)`, a face counts as hidden by default when a blur or a doubtful zone covers at least 95 percent of its box with the engine's own margin. The 2.2 defaults blur only the doubtful items that are placed like a plate or a face ([How it works](concepts/how-it-works.md#doubtful-items)).

**Why some faces are missed.** Of the {misses_headline.careful.misses} faces Careful misses on the photos never used to choose a number, {misses_headline.careful.misses_under_20_px_pct}% are under 20 px tall and {misses_headline.careful.misses_under_40_px_pct}% under 40 px, mostly people far from the camera; on faces at least 40 px tall it hides {misses_headline.careful.hidden_at_least_40_px_pct}%. On the 30 blind photos, {misses_headline.careful.blind_misses_turned_or_covered} of the {misses_headline.careful.blind_misses} faces it misses are turned away or partly covered. The breakdown is in the repository, `docs/MISSES.md`.

Every figure with its method, the comparison with an open reference detector and the earlier measurements are in the repository, `docs/MEASURED.md`. The photos and clips are public, and `bench/SOURCES.csv` lists them.

## Limits

- **A miss is silent.** A face or a plate the detectors do not find is neither blurred nor listed, and nothing in the output or the report says it was there. Play the output and look, and use **Add missed area** for what you find.
- **Faces that are small, turned, blurred or partly covered** are missed more often than a frontal face in good light. A photograph on a wall or a face on a screen is treated like any other face: **Review** lets you switch it off.
- **Plates are read as text on the vehicles the vehicle detector finds.** A plate on anything else, on a vehicle too small to be searched, or scored low by the text detector is not blurred. A doubtful plate in a video is left visible by default: review the plates of a video yourself.
- **The second look has a budget.** On a video it can leave many places unchecked. They keep the first pass's result, and the banner after the analysis says how many.
- **Tested on Linux only.** The Studio ran on Ubuntu 22.04 with Python 3.10, in Chrome and Firefox 157, and the engine and the tests on two more Ubuntu machines. Windows, macOS and Safari were not tried: `packaging/windows/` never ran on Windows, and neither did the search for a browser to print the PDF report. The optional window ran under a virtual screen only, and ending the Studio with `kill` does not close it. On ARM, only three tests of the privacy audit were run.

Known issues in 2.2.0:

- A photo analysed in Careful shows no progress until its first pass ends, and **Stop** cannot cut that pass short.
- The progress line of a video does not say whether the second look is running. A photo's does.
- The interface texts of the seven languages were checked by back-translation only (each translated back to English without its source and found faithful), not by native speakers.
