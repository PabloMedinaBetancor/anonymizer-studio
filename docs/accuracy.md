# Accuracy and limits

How often Anonymizer Studio hides the faces and plates it is meant to blur, what each figure was measured on, and what the product does not do. Read it before you decide whether the Studio fits your footage. The figures are read from the results files by a script; none is typed here.

## The figures

| What was measured | Result | Condition |
|---|---|---|
| Faces in photos hidden by default, Careful | 76.51% of 1222 (Wilson interval 74.06-78.81) | 2.1's settings, on photos never used to choose a number: the 30 blind photos of the next row and the sealed half of WIDER FACE |
| Faces in 30 blind photos of the kind a real-estate agent takes, Careful | 60.26% of 78 (Wilson interval 49.16-70.39) | 2.1's settings. The figure to quote for listing photos. There is no blind figure for the 2.2 defaults |
| Faces in photos, the 2.2 defaults | 97.1% of 623 faces in 91 photos | The development part, which the rules were fitted on and whose labels started from the detector's own proposals: in-sample and optimistic |
| Readable plates in photos, the 2.2 defaults | 77.6% of 58 plates in 74 photos | The development part: in-sample |
| Plates of the OpenALPR EU set, the 2.2 defaults | 105 of 108 hidden | Never used to choose a number; the boxes are OpenALPR's own |
| Readable plates of 60 street photos, the 2.2 defaults | 8 of 16 hidden | Never used to choose a number. Too few plates for a rate, so none is quoted |
| Faces in video covered at least 80 percent, Careful | 115 of 137 | Three sealed clips, one of them a crowd that holds most of the faces. In video the second look did not measurably add to Fast (0.73 points, not significant). Plates in video were not measured |
| Photos without people on which something was hidden, Careful | 20 per 100 (8 of 40 photos); Fast 7.5 | 2.1's settings. This rate is not claimed to be low; the 2.2 defaults trade it for some plates and faces left visible |
| Seconds per photo for face detection alone, median | Careful 0.56, Fast 0.24 | The 153 sealed face photos, one run on the development laptop (an Intel Core i7-8550U), which was not idle: read them as indicative. With plates a photo takes seconds, and a video much more. The Studio shows an estimate while it analyses, and Careful costs more than Fast |

In the rows with 2.1's settings, `Options(unsure="blur", fit="loose", plate_band=True)`, a face counts as hidden by default when a blur or a doubtful zone covers at least 95 percent of its box with the engine's own margin. The 2.2 defaults blur only the doubtful items that are placed like a plate or a face ([How it works](concepts/how-it-works.md#doubtful-items)).

**Why some faces are missed.** Of the 287 faces Careful misses on the photos never used to choose a number, 58.89% are under 20 px tall and 87.11% under 40 px, mostly people far from the camera; on faces at least 40 px tall it hides 87.91%. On the 30 blind photos, 21 of the 31 faces it misses are turned away or partly covered. The breakdown is in the repository, `docs/MISSES.md`.

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
