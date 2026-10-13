# Measured results

Every measurement behind the figures of the documentation: how often Anonymizer Studio finds the faces and plates it is meant to blur, how long it takes, and what each figure was measured on. The short version, with the limits, is [Accuracy and limits](accuracy.md). Every figure here carries its sample size, its interval and its condition, and the figures that were fitted on their own data are kept apart from the ones that were not.

## In short

The figures below are read from the results files by a script. They are not typed here.

- **Faces in photos, 2.1's settings, on parts that were never used to choose a number.** The Careful level hides 76.51% of 1222 faces by default (Wilson interval 74.06-78.81). On the 30 blind photos of the kind a real-estate agent takes, it hides 60.26% of 78 faces (Wilson interval 49.16-70.39). That is the figure to quote for listing photos. These are the only blind face figures, and they were measured with 2.1's settings, not the 2.2 defaults. There is no blind figure for the 2.2 defaults.
- **Faces in photos, the 2.2 defaults, on the development part.** 97.1% of the 623 faces in 91 photos are hidden. The rules were fitted on this part, and its labels started from the detector's own proposals, so this figure is in-sample and optimistic.
- **Plates in photos, the 2.2 defaults.** On the development part, which the rules were fitted on, 77.6% of the 58 readable plates in 74 photos are hidden. On two sets that were never used to choose a number, 105 of 108 plates of the OpenALPR EU set are hidden, and 8 of 16 readable plates of 60 street photos, which is too few plates for a rate. No recall is quoted for street plates.
- **Faces in video.** On three sealed clips, Careful covers 115 of 137 faces at least 80 percent. In video the second look did not measurably add to Fast (0.73 points at 80 percent cover, not significant). Plates in video were not measured.
- **What it does not do well.** It misses small, blurred, turned and partly covered faces, and a miss is neither blurred nor listed. The rate at which it blurs something that is not a face or a plate is not low with 2.1's settings, and the 2.2 defaults trade it for some plates and faces left visible. The sections below have the figures, and [Accuracy and limits](accuracy.md#limits) has the rest.
- **Time.** Careful costs more than Fast, and a video costs much more than a photo. The tables give the time with the machine, the settings and the clip.

## How the figures were measured

Every figure below is filled in from `bench/results/*.json` by `scripts/fill_numbers.py`, which refuses any metric that its results file marks as not publishable. The photos and clips are public (Wikimedia Commons, and the OpenALPR benchmark for plates); `bench/SOURCES.csv` gives the page, author and licence of each. Where a recall is quoted with its 95-percent interval (Wilson), the interval comes from the same results file.

Two laptops were used, and each figure says which. The development laptop has an Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz. The second laptop has an Intel(R) Core(TM) i3-6006U CPU @ 2.00GHz. A speed belongs to the machine it ran on and will differ on yours.

The sealed read and the table of the 2.2 defaults are the current measurements. The first measurements ran on an older plate pipeline and on the older face photos, and are kept at the end of the page as history. 2.1's text path blurred all text on a vehicle, so the false-blur rows of the retired plate model are not repeated: that measure is not the same one.


## The sealed read of the detection levels (v2)

The comparison of the detection levels ran once, on the product as frozen, on photos and clips whose labels had been kept unseen until then: 153 face photos and 34 plate photos. The figures come from `bench/results/sealed-v2.json`, and `bench/` has the details of the method. Four things weaken it, and they are said here so that the figures are not read as stronger than they are: the images (not their labels) were touched by two runs that kept nothing, namely timing runs of the heavy model on 5 sealed photos and a stopped run that decoded frames of the three sealed clips; the person who labelled the 30 blind photos had seen them; the band variant, the crop side and the colour order were chosen or checked on the development part; and a first attempt at freezing the product was voided before any read. The run was on the development laptop, on the balanced power profile.

A face counts as hidden by default when a blur or a doubtful zone covers at least 95 percent of its box with the engine's own margin, with `unsure="blur"`. So these figures are of 2.1's defaults, `Options(unsure="blur", fit="loose", plate_band=True)`; what the 2.2 defaults hide is in the section after the tables.

The independent parts are the 30 blind photos of the kind a real-estate agent takes and the sealed half of the WIDER FACE photos, 1222 faces in all. They are the headline. The older face photos, whose labels started from a detector's proposals, are reported apart and only in the "all four parts" column.


| Level | Faces hidden by default, the independent parts (blind real-estate-kind photos and the sealed half of WIDER) | The 30 blind photos alone (their faces are not independent: the interval by photo is wider) | All four parts (adds the older face photos) | False hidden, per 100 photos without people | Seconds per photo, median |
|---|---|---|---|---|---|
| Fast | 70.87% (866 of 1222; Wilson 68.26-73.35) | 58.97% (46 of 78; Wilson 47.89-69.22) | 78.25% | 7.5 (3 of 40 photos; Wilson 2.58-19.86) | 0.24 |
| Control: the light model zoomed on the same zones | 74.96% (916 of 1222; Wilson 72.45-77.31) | 60.26% (47 of 78; Wilson 49.16-70.39) | 81.83% | 22.5 (9 of 40 photos; Wilson 12.32-37.5) | 0.35 |
| Careful | 76.51% (935 of 1222; Wilson 74.06-78.81) | 60.26% (47 of 78; Wilson 49.16-70.39) | 82.96% | 20 (8 of 40 photos; Wilson 10.5-34.76) | 0.56 |
| Reference `deface` (CenterFace) at the cut chosen on dev | 36.66% (448 of 1222; Wilson 34.01-39.4) | 28.21% (22 of 78; Wilson 19.43-39.03) | 44.18% | 7.5 (3 of 40 photos; Wilson 2.58-19.86) | 0.24 |
| Reference `deface` at its own default cut | 53.93% (659 of 1222; Wilson 51.13-56.71) | 38.46% (30 of 78; Wilson 28.45-49.56) | 60.31% | 52.5 (21 of 40 photos; Wilson 37.5-67.06) | not timed |


| Paired comparison, all faces | Faces hidden only by the first / only by the second | Exact McNemar p | Gain per 1,000 faces (bootstrap by photo, 95 percent) |
|---|---|---|---|
| Careful over fast | 83 / 0 | < 0.0001 | 47.1 (28.8-75.9) |
| Careful over the control | 36 / 16 | 0.00779 | 11.4 (3.5-22.6) |
| Careful over the reference (cut chosen on dev) | 690 / 7 | < 0.0001 | 387.8 (249.4-486.9) |


What this table says, plainly:

- **Careful** hides 76.51% of the faces on the independent parts, and only 60.26% (47 of 78 faces in 30 photos; Wilson 49.16-70.39, by photo 42.86-71.88) on the 30 blind photos of the kind a real-estate agent takes. That is the figure to quote for this case. The other independent part, the sealed half of WIDER, gives 77.62% (888 of 1144; Wilson 75.12-79.94); the older face photos, with the caveat above, 97.59%.
- **The second look pays**: careful hides faces fast misses and loses none (p below 0.0001), and it also beats the light model zoomed on the same zones (small gain, p 0.00779).
- **The false hidden rate is not claimed to be low**: on photos without people careful hides something on 20 per 100 photos (fast 7.5). `unsure="keep"` leaves doubtful items visible and listed, and is the way to avoid these.
- **Time**: the median of the per-photo ratio careful over fast is 2.054 (bootstrap 1.855-2.251), paired inside one run on one machine. The machine was not idle when the read started (a load average of 1.09), so read the times as indicative.
- **`deface`, the open reference, does worse at either cut**: it hides fewer faces than every level of this product and, at its default cut, blurs something on half of the photos without people. It stays in the table.
- Fast is the cheapest and hides the fewest of the two levels the Studio offers; no claim is made about EgoBlur, which was not run.


**Experimental, not offered: Deep.** The Python API still has `detection="deep"` (the heavy model over the whole picture, in tiles; `heavy_only=True` leaves the light class models out of the result). The Studio does not offer it: it is too slow and too sensitive for publishing work. Measured in the same sealed read, kept here for the record and not part of the headline: on the independent parts Deep hides 81.51% of the faces (996 of 1222; Wilson 79.23-83.58) and 62.82% on the 30 blind photos, hides something on 22.5 per 100 photos without people, and takes 3.82 s per photo (median; careful 0.56 s); over careful it hides 78 faces more and loses 13 (exact McNemar p below 0.0001). With the heavy model alone (`heavy_only`) it hides 74.47% (910 of 1222). Memory with every model loaded: deep 906.8 MB on the development laptop and 944.8 MB on the second laptop (the project's own ceiling for Deep was 1600 MB).


**Why some faces are missed.** The faces that are not hidden are mostly small, blurred, turned or partly covered ones. Of the 287 faces Careful misses on the independent parts, 58.89% are under 20 px tall and 87.11% under 40 px (small faces, mostly people far from the camera); on faces at least 40 px tall it hides 87.91%. Fast misses 356, 84.83% of them under 40 px.
Distance is not the whole story: in WIDER a tiny face is usually also a heavily blurred one, and Careful hides 93.64% of the WIDER faces with no heavy blur, heavy occlusion or odd pose, against 61.94% of those with one. On the 30 blind photos, 21 of the 31 faces Careful misses are turned away or partly covered, and only 3 are small and nothing else.
Nothing here measures whether a missed face could still be recognised. This is a post-hoc breakdown of the sealed read with unchanged code; it changes no figure above. Tables and method: [docs/MISSES.md](MISSES.md).


## Plates and video, sealed

| Plates, sealed photos | Plates | Plate hidden with 2.1's settings (text path and band) | Text regions alone | Old YOLOv9-t model (reference path only, not shipped) | 2.1's settings only / old model only | The band adds: with band only / without only | Band cost: share of the picture as a fraction, so 0.04 is 4 percent (mean over the 34 photos; banded photos; at most) |
|---|---|---|---|---|---|---|---|
| full size | 51 | 100% (51; Wilson 93-100 %) | 44 | 45 | 6 / 0 (p 0.03125: 117.6 per 1,000 plates) | 7 / 0 (p 0.01562) | 0.04139 (banded photos: 0.05413; at most 0.15964); 0 bands on the 5 photos with no searchable vehicle |
| half size | 51 | 86.27% (44; Wilson 74.28-93.19 %) | 38 | 40 | 8 / 4 (p 0.3877: 78.4 per 1,000 plates) | 6 / 0 (p 0.03125) | 0.04863 (banded photos: 0.07515; at most 0.17889); 0 bands on the 7 photos with no searchable vehicle |
| a third of the size | 51 | 64.71% (33; Wilson 50.99-76.37 %) | 27 | 33 | 5 / 5 (p 1: 0 per 1,000 plates) | 6 / 0 (p 0.03125) | 0.06801 (banded photos: 0.10054; at most 0.22449); 0 bands on the 8 photos with no searchable vehicle |

Plates: at full size 2.1's settings hide every plate and the old model misses some (p 0.03125); at half size the two are not significantly different and at a third of the size they match. The band is what adds the plates at every size, and its cost is in the last column. The same settings on the earlier plate sets are in the earlier measurements at the end of the page, in-sample for the band (fitted on them): the sealed photos are the out-of-sample figure. The plates hidden by the 2.2 defaults are in the section after this one.


| Video, three sealed clips | Faces covered at least 80 percent | Wilson (bootstrap by clip) | Covered at least 50 percent |
|---|---|---|---|
| Fast | 114 of 137 = 83.21% | 76.07-88.54 % (82.11-88.89 %) | 84.67% |
| Careful | 115 of 137 = 83.94% | 76.88-89.15 % (81.05-90.91 %) | 85.4% |

Video: careful over fast does not hold (0.73 points at 80 percent cover, p 1, so the gain could be chance). **In video the second look did not measurably add to Fast.** The clips are the result, and the crowd clip dominates the pool.


Memory, measured on both laptops: Careful peaks at 1040.3 MB on the development laptop and 1020.3 MB on the second laptop, below the ceiling of 1200 MB that the project set for itself. The measure is one photo of the older face photos at 2,048 px on the long side, every model loaded, one detection level per process (`bench/results/memory-v2.json`).



## What the 2.2 defaults hide

2.1's defaults blurred every text region on a vehicle, the plate-position band and every doubtful face, with wide margins and a wide soft edge: on a lot with
parked cars a blob over the side of a car and one over a bare wall. 2.2 hides the plate itself and the face itself and leaves the rest of the picture as it was
(`Options.unsure="placed"`, `fit="tight"`, no plate band; the rule is on [How it works](concepts/how-it-works.md#doubtful-items)).
A text region on a vehicle is blurred by default when it scored high, sits where a plate can be and is not a word: a second model, the plate reader, reads
the region, takes out what reads as a name or a sign ("TAXI", "RENAULT") and brings in a plate that the position rule had rejected. A doubtful face is blurred
when the heavy model of the second look, or the first sweep, is sure of it and it is not tiny.
The cost is the other side of the same choice: what the detectors read as writing on a vehicle but that is not shaped, placed and read like a plate is listed in Review and left visible,
and so are some small, blurred or far plates that the text detector scores low or does not find. `bench/bench_subtle.py` measures the rules side by side on the dev part
of the labelled photos (never the sealed part): the plates and faces hidden, the false regions per 100 photos, and the share of the picture that the blur really
changes outside the labelled objects; its result is `bench/results/subtle.json`. The rows are 2.1.10's defaults, the first round of 2.2 (margins 0.06 / 0.10, no reader, no
agreement for the faces) and the shipped 2.2, all run by the same code.

| Dev part, defaults of | Readable plates hidden | All plates hidden | Plate false regions per 100 photos | Picture blurred outside the plates | Faces hidden | Face false regions per 100 photos |
|---|---|---|---|---|---|---|
| 2.1.10 (`unsure="blur"`, loose, band) | 98.3% | 93.5% | 1513.5 | 17.026% | 97.4% | 23.1 |
| 2.2, first round | 74.1% | 43.5% | 66.2 | 0.737% | 94.1% | 18.7 |
| 2.2 (this version) | 77.6% | 45.4% | 39.2 | 0.615% | 97.1% | 19.8 |

Plates: 74 photos, 108 plates, 58 of them readable. Faces: 91 photos, 623 faces. The rules were fitted on this part. Two sets that are outside the split and
were never used to choose a number are the check (`bench/bench_subtle.py --sets streets,eu`); false regions are measured on the street photos only (the EU set holds plates that its
truth does not annotate):

| Check, defaults of | Street photos (60): readable plates hidden | Street photos: false regions per 100 | OpenALPR EU (108 photos): plates hidden |
|---|---|---|---|
| 2.1.10 | 16 of 16 | 713.3 | 108 of 108 |
| 2.2, first round | 8 of 16 | 10 | 102 of 108 |
| 2.2 (this version) | 8 of 16 | 6.7 | 105 of 108 |

The old rule is `Options(unsure="blur", fit="loose", plate_band=True)`.


## Machines, and what each truth can say

Each figure names its machine. The earlier speed rows ran on the second laptop (Intel(R) Core(TM) i3-6006U CPU @ 2.00GHz, 4 cores, 3.3 GB RAM, Linux, one process). The sealed read, the 2.1 settings on the earlier plate sets and the table of the 2.2 defaults ran on the development laptop, on the balanced power profile. The memory was measured on both.

What each truth can and cannot say (the full account is in `bench/truth/faces/README.md` and
`bench/truth/plates/README.md`):

The boxes are also kept as a separate dataset, labels only and no images (`docs/truth-dataset/`): the label files with their
SHA-256, each image's page, licence and SHA-256, the method and the licence of the labels.


- **Plates, OpenALPR EU.** The boxes are OpenALPR's own annotations, so the truth is independent of this project. Every
  plate in it is readable; a photo can show other plates upstream did not annotate, so false blurs are not counted on
  this set. The plate figures of this set quoted anywhere are of these 108 plates.
- **Plates, streets.** 16 readable plates in 60 photos. A
  second labeller who had not seen the first truth redid 30 of the photos; the two agreed on a minority of the plates,
  and the truth used is the union of both. That is enough to count false blurs over the 60
  street photos, and too few plates with too weak a truth to quote a recall, so none is quoted.
- **Faces, photos.** The truth was built from detector proposals at a low threshold and checked blind on 15 photos: an
  optimistic bound. Faces that YuNet cannot find at any scale and that the reviewer did not notice are missing from the
  count. The blind check covered only photos with few faces; crowds were not relabelled. The truth was also corrected
  after the first benchmark run: of the 113 labelled faces that no blur covered then, 56 were judged wrong or not
  defensible (backs of heads, hands over faces, faces behind glass) and turned into `ignore` boxes. A blur over an
  `ignore` box counts neither as a hit nor as a false blur; the engine blurs 200 of the
  1157 `ignore` boxes. The faces of one photo are not independent, so the 95-percent interval is
  narrower than the real uncertainty.
- **Faces, video.** The earlier sample (22 faces in one street clip, nearly all profiles in deep shadow, one reviewer) is not published as a recall. The sealed clips are: three clips with 137 labelled faces, one crowd clip that holds most of them, and a second look that did not measurably add to Fast (the table above).
- **Plates, video.** Not measured.


## Earlier measurements

These ran before the sealed read, on an older plate pipeline (with the YOLOv9-t model, which is no longer shipped) and on the older face photos. They are not compared with the sealed read, because the truth, the product and the rule differ. They are history, not the figures to quote.

| What (earlier measurements) | Result | Data |
|---|---|---|
| Plates hidden with 2.1's settings (text path and band), OpenALPR EU set, at full size / half size / a third of the size | 99.07% / 100% / 100% | 108 readable plates (107/108 at full size; 95-percent interval at full size: 94.94-99.84 %), boxes annotated by OpenALPR, not by us. In-sample for the band: it was fitted on these two earlier sets |
| The same settings, readable plates of the street photos | 16/16 (too few plates for a rate) | the street photos of the earlier plate set; in-sample for the band; the sealed plate photos above are the out-of-sample figure |
| Faces in photos blurred automatically | 77.71% (903 of 1162; 95-percent interval 75.2-80 %, narrower than the real uncertainty, see below) | Commons photos of crowds, markets, groups, streets |
| Faces in photos blurred or marked Not sure | 95.09% | same faces |
| Faces in photos, counting the ellipse that is really drawn | 75.9% | same faces |
| Faces under 24 px wide, blurred automatically | 64.8% of 304 | same set |
| False automatic face blurs, per 100 photos without people (YuNet; the sealed read above has 20 per 100 in careful and 7.5 in fast) | 0 (0 in 71 photos); with none seen, the rule of three only puts the share of such photos with a false blur under 4.2 % | 71 photos of statues, paintings, mannequins, car grilles |
| Gap filling between detections, with simulated perfect recovery: camera motion / straight line / hold | 97.06% / 96.55% / 95% of object-frames covered | synthetic video with exact truth, 12613 object-frames, 10 seeds |
| Processing time, analyze and render, per second of 1080p video, on a clip with nothing to blur | 5.57 s (4.21 analyze, 1.37 render) | faces/video/hongkong.webm (1920x1080, Wikimedia Commons, bench/SOURCES.csv): 0 items, 0 % of frames with a blur, so this is the floor of a recording with nothing to hide; faces and plates, Standard, default margins; analyze + render, one process |
| Processing time, analyze and render, per second of 720p video, on a clip blurred in most frames | 4.49 s (3.59 analyze, 0.89 render) | faces/video/preciados.ogv (1280x720, Wikimedia Commons, bench/SOURCES.csv): 35 items, 93.8 % of frames with a blur; same settings |
| Peak memory after analysing, 1080p clip / 720p clip (earlier video engine; the sealed read's memory is above) | 235 MB / 288 MB | models loaded |
| A second render after one review change, against the whole run, on the 720p clip | 4.94 times faster (0.91 s per second of video) | no detector runs in it |
| Analysis time of Maximum against Standard (face detector only) | Maximum takes 5.09 times as long | face detector only, two public clips, 52 frames labelled by hand |
| Time per photo, faces and plates together, analyze and write | 1948.9 ms (1862.4 ms of it analysis) | 60 street photos, 1600 to 1920 px wide; faces and plates together, Standard, through the public API (analyze, then render) |
| Stop: from a stop request to the engine stopping (`Cancelled` raised) | at most 0.18 s, median 0.04 s | 20 stops from another thread at arbitrary moments, on the 1080p clip; the requirement is 2 s on a free machine |


The open reference for a comparison is `deface` (CenterFace detector, MIT). **EgoBlur was not run**, so nothing here is a
claim against it, and a reference that does better than this product on a measure stays in the table.
