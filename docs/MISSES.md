# Why some faces are missed

The question a client asks: of the faces the Studio did not blur, why did it miss them? Were they too far away?

This page answers it from the sealed read of the faces, face by face. Every table and every sentence with a number on it is written by
`bench/miss_breakdown.py` from the files under `bench/results/` (`miss-breakdown-log.json`, `miss-breakdown.json`, `miss-blind-tags.json`);
a test fails if this page and those files disagree.

These figures are of the v2.1.1 engine with 2.1's settings, `unsure="blur"`: every doubtful zone counts as blurred. With the 2.2 defaults a doubtful face that is not placed is left visible, so more faces stay visible than this page counts.

## The short answer

<!-- BEGIN answer -->
- **Careful** (the default) leaves 287 of 1222 faces on the independent parts without the 95 % cover the read asks for. Of those 287, **58.9 % are under 20 px tall, 76.7 % under 30 px and 87.1 % under 40 px**.
- On faces at least 20 px tall (743 faces) Careful hides 84.1 %; at least 40 px (306), 87.9 % (Wilson 83.8-91.1); at least 80 px (107), 92.5 %. Under 20 px (479 faces) it hides 64.7 %.
- **Fast**: of its 356 misses, 52.0 % are under 20 px and 84.8 % under 40 px; on faces at least 40 px tall it hides 82.3 %.
<!-- END answer -->

So **mostly yes, the faces that are missed are mostly small ones**, and the smaller the face the likelier the miss. But distance is not
the whole story, and the page below says what else is going on:

<!-- BEGIN hindrance-summary -->
- WIDER records for each face how blurred (clear/normal blur/heavy blur), how occluded and what pose it has. Call a face one **with a hindrance** if it is heavily blurred, heavily occluded or in an atypical pose: 578 of the 1144 WIDER faces on the sealed part are.
- Careful hides 61.9 % of the faces with a hindrance and 93.6 % of the faces without, and 220 of its 256 misses on WIDER (85.9 %) are faces with a hindrance.
<!-- END hindrance-summary -->

<!-- BEGIN blind-summary -->
- Of the 31 faces Careful misses on the blind photos, 12 are mainly turned (profile, three-quarter, looking down, the back of the head), 9 partly hidden, 6 mainly lost in a dark scene, 1 blurred and only 3 just small; 19 of the 31 sit in a dark or grainy scene.
- The blind faces are small (69 of 78 are under 40 px) and Careful hides 59.4 % of them, against 95.4 % for the WIDER faces of 20-40 px with no hindrance recorded: on these photos the small size comes with dim light, turned heads and people in front of people, so size is not the only thing at work.
<!-- END blind-summary -->

<!-- BEGIN loose-summary -->
- Most misses are faces with nothing blurred on them: Careful leaves 188 of its 287 misses (65.5 %) with no blur at all and the other 99 partly blurred. Counting a face as hidden at 80 % of its box instead of 95 % would move Careful from 76.5 % to 80.0 %, and at half of its box to 81.8 %.
<!-- END loose-summary -->

What this page does **not** say: it does not measure whether a person could still be recognised from a missed face. The smaller
the face, the fewer pixels a viewer has to recognise it with, but no recognition test was run, so that is not claimed. Read the
tables as "how big and how hard to see are the faces that were missed", not as "these misses do not matter".

## What this is, and what it is not

- **A post-hoc breakdown of the sealed read, with unchanged code.** The sealed read (`bench/results/sealed-v2.json`, 2026-10-04) stored totals, not which
  face each level missed. For this page the same measurement code (`bench/sealed_measure.py`, the loop of `faces_v2`) was run again over the same sealed photos
  and the same truth, with the engine as it is in v2.1.1 (byte-identical to the engine of the sealed read), for the levels Fast, Careful and
  Deep, and one record was kept for every labelled face: its size, which level hid it, and the zone that came closest (`bench/results/miss-breakdown-log.json`). The records of the WIDER faces carry no photo file name, no box and no picture size, only opaque ids.
- **The totals reproduce exactly.** A breakdown that did not add up to the published figures would describe something else, so the log is checked
  against `sealed-v2.json` (each level and part, the faces hidden by default, and the shares hidden with certainty and as drawn ellipses). They are equal:

<!-- BEGIN reproduction -->
| Faces hidden by default: this log / `sealed-v2.json` | Fast | Careful | Deep |
|---|---|---|---|
| independent parts (blind + WIDER) | 866 of 1222 / 866 of 1222 | 935 of 1222 / 935 of 1222 | 996 of 1222 / 996 of 1222 |
| blind photos | 46 of 78 / 46 of 78 | 47 of 78 / 47 of 78 | 49 of 78 / 49 of 78 |
| WIDER, sealed half | 820 of 1144 / 820 of 1144 | 888 of 1144 / 888 of 1144 | 947 of 1144 / 947 of 1144 |
| set a | 512 of 539 / 512 of 539 | 526 of 539 / 526 of 539 | 530 of 539 / 530 of 539 |
<!-- END reproduction -->

- **It changes no published figure, no code and no threshold.** Nothing in the engine, the measurement or the README figures was touched because of
  what is below. This is a description of the misses, not a new result to put next to the old ones.
- **"Hidden" is the read's rule**: a face counts as hidden when the blur (or a "Not sure" zone, blurred with 2.1's settings) covers at least 95 % of its labelled box, with the
  engine's margin. The independent parts, 1222 faces in the 30 blind photos and the sealed half of WIDER FACE, are the ones analysed; set a (the older photos, whose
  truth started from a detector) is in the log and in the totals check only.
- **Sizes are the height of the labelled face box in pixels of the photo as stored**, and the same height as a share of the picture height. The engine looks at the
  picture at several scales, so a face of a given size in a large photo is not the same job as in a small one; both columns are given.
- **The intervals are Wilson 95 % intervals and treat the faces as independent, which faces of one photo are not** (the crowd photos hold dozens of faces).
  They are indications of how much a figure could move, not guarantees; the by-photo intervals of the accuracy page are the ones to quote.
- **WIDER attributes** (blur, occlusion, pose) come from the WIDER FACE annotation file, read from the local copy of its annotation archive; nothing was fetched. WIDER FACE is research-only data and its annotation is not redistributed: the log keeps the five attribute codes of each face, because the tables below are computed from them, and nothing that says which photo or where (opaque ids `wider#0001` ..., no file name, no box, no picture size, rows in an order taken from their own values).
- **Never published: crops or pictures of the missed faces.** WIDER and the third-party photos are test data; this page has numbers only. The by-eye tags of the blind
  photos are judgements made by looking at each missed face enlarged in its photo (`bench/results/miss-blind-tags.json`), and no crop was kept.

## How big are the missed faces

Independent parts, 1222 faces. Face height in pixels:

<!-- BEGIN size-px -->
| Face height (px) | Faces | Fast hides | Careful hides | Deep hides | Share of all of Careful's misses |
|---|---|---|---|---|---|
| 10-20 px | 479 | 61.4 % (294 of 479; 56.9-65.6) | 64.7 % (310 of 479; 60.3-68.9) | 74.1 % (355 of 479; 70.0-77.8) | 58.9 % |
| 20-40 px | 437 | 73.2 % (320 of 437; 68.9-77.2) | 81.5 % (356 of 437; 77.5-84.8) | 83.5 % (365 of 437; 79.8-86.7) | 28.2 % |
| 40-80 px | 199 | 81.4 % (162 of 199; 75.4-86.2) | 85.4 % (170 of 199; 79.8-89.7) | 88.9 % (177 of 199; 83.8-92.6) | 10.1 % |
| 80 px and over | 107 | 84.1 % (90 of 107; 76.0-89.8) | 92.5 % (99 of 107; 85.9-96.2) | 92.5 % (99 of 107; 85.9-96.2) | 2.8 % |
<!-- END size-px -->

The same, with the face height as a share of the picture height:

<!-- BEGIN size-frac -->
| Face height (share of the picture height) | Faces | Fast hides | Careful hides | Deep hides | Share of all of Careful's misses |
|---|---|---|---|---|---|
| 1-2 % | 308 | 54.2 % (167 of 308; 48.6-59.7) | 57.1 % (176 of 308; 51.6-62.5) | 65.9 % (203 of 308; 60.5-71.0) | 46.0 % |
| 2-4 % | 431 | 71.5 % (308 of 431; 67.0-75.5) | 78.2 % (337 of 431; 74.0-81.8) | 83.3 % (359 of 431; 79.5-86.5) | 32.8 % |
| 4-8 % | 311 | 80.7 % (251 of 311; 76.0-84.7) | 86.8 % (270 of 311; 82.6-90.1) | 89.4 % (278 of 311; 85.5-92.3) | 14.3 % |
| 8 % and over | 172 | 81.4 % (140 of 172; 74.9-86.5) | 88.4 % (152 of 172; 82.7-92.3) | 90.7 % (156 of 172; 85.4-94.2) | 7.0 % |
<!-- END size-frac -->

Cut-offs ("of the misses, how many are under X px; on faces at least X px tall, what is hidden"):

<!-- BEGIN cuts -->
| Cut | Faces under it | Faces at least that tall | Fast: misses under the cut | Fast hides at least that tall | Careful: misses under the cut | Careful hides at least that tall |
|---|---|---|---|---|---|---|
| 20 px | 479 | 743 | 52.0 % of 356 | 77.0 % (572 of 743) | 58.9 % of 287 | 84.1 % (625 of 743) |
| 30 px | 719 | 503 | 72.2 % of 356 | 80.3 % (404 of 503) | 76.7 % of 287 | 86.7 % (436 of 503) |
| 40 px | 916 | 306 | 84.8 % of 356 | 82.3 % (252 of 306) | 87.1 % of 287 | 87.9 % (269 of 306) |
| 60 px | 1062 | 160 | 91.6 % of 356 | 81.2 % (130 of 160) | 93.7 % of 287 | 88.8 % (142 of 160) |
| 80 px | 1115 | 107 | 95.2 % of 356 | 84.1 % (90 of 107) | 97.2 % of 287 | 92.5 % (99 of 107) |
<!-- END cuts -->

## Distance is not the whole story

No. In WIDER, a face is small and blurred at the same time more often than not: tiny faces are the ones the annotators call heavily blurred. So the
size table above mixes two things. Splitting WIDER faces by whether the annotation records a hindrance (heavy blur, heavy occlusion, atypical pose) and by size:

<!-- BEGIN hindrance -->
| WIDER faces | Face height | Faces | Fast hides | Careful hides | Deep hides |
|---|---|---|---|---|---|
| no hindrance recorded | under 20 px | 113 | 90.3 % (102 of 113; 83.4-94.5) | 90.3 % (102 of 113; 83.4-94.5) | 92.9 % (105 of 113; 86.7-96.4) |
| no hindrance recorded | 20-40 px | 218 | 90.4 % (197 of 218; 85.7-93.6) | 95.4 % (208 of 218; 91.8-97.5) | 94.5 % (206 of 218; 90.6-96.8) |
| no hindrance recorded | 40 px and over | 235 | 91.5 % (215 of 235; 87.2-94.4) | 93.6 % (220 of 235; 89.7-96.1) | 94.5 % (222 of 235; 90.8-96.7) |
| with a hindrance | under 20 px | 365 | 52.3 % (191 of 365; 47.2-57.4) | 56.7 % (207 of 365; 51.6-61.7) | 68.2 % (249 of 365; 63.3-72.8) |
| with a hindrance | 20-40 px | 151 | 55.6 % (84 of 151; 47.7-63.3) | 71.5 % (108 of 151; 63.9-78.1) | 78.2 % (118 of 151; 70.9-84.0) |
| with a hindrance | 40 px and over | 62 | 50.0 % (31 of 62; 37.9-62.1) | 69.3 % (43 of 62; 57.0-79.4) | 75.8 % (47 of 62; 63.9-84.8) |
<!-- END hindrance -->

The same by single attribute (WIDER's own labels, faces of the sealed half):

<!-- BEGIN attributes -->
| WIDER attribute | Value | Faces | Careful hides | Share of Careful's misses on WIDER |
|---|---|---|---|---|
| blur | clear | 172 | 93.6 % (161 of 172) | 4.3 % |
| blur | normal blur | 519 | 87.9 % (456 of 519) | 24.6 % |
| blur | heavy blur | 453 | 59.8 % (271 of 453) | 71.1 % |
| occlusion | none | 673 | 87.4 % (588 of 673) | 33.2 % |
| occlusion | partial | 210 | 73.8 % (155 of 210) | 21.5 % |
| occlusion | heavy | 261 | 55.6 % (145 of 261) | 45.3 % |
| pose | typical | 1114 | 78.1 % (870 of 1114) | 95.3 % |
| pose | atypical | 30 | 60.0 % (18 of 30) | 4.7 % |
<!-- END attributes -->

What this means plainly: a small face that is otherwise clear is found most of the time. The faces that are lost are the small ones that are also blurred, turned
away or partly covered, and a heavily blurred or covered face is lost at any size. Size and blur cannot be fully separated on this data, because WIDER's
blur label follows from how few pixels a face has; read the two tables together.

## Missed faces and partly blurred faces

A miss is one of two things: no face zone was put on the face at all, or a zone was put on it but it covered less than 95 % of the labelled box (a box that is a bit too
small or off to one side). With 2.1's settings (`unsure="blur"`) the engine's "Not sure" zones are blurred, so a face found as unsure is never left visible because of that verdict:
the column "Misses that were unsure and left unblurred" is zero by construction, and the unsure zones appear as partly covered faces or as faces hidden only as "unsure".

<!-- BEGIN causes -->
| Level | Misses | No face zone touches the face | A face zone touches it but covers under 95 % | Misses that were unsure and left unblurred | Faces hidden only as "unsure" |
|---|---|---|---|---|---|
| Fast | 356 | 245 (median height 15 px) | 111 (median height 26 px) | 0 | 186 of 866 |
| Careful | 287 | 188 (median height 14 px) | 99 (median height 23 px) | 0 | 174 of 935 |
| Deep | 226 | 113 (median height 14 px) | 113 (median height 24 px) | 0 | 270 of 996 |
<!-- END causes -->

<!-- BEGIN loose -->
| Level | Hidden by the read's rule (at least 95 %) | At least 80 % of the face blurred | At least 50 % blurred | Misses with nothing blurred on the face |
|---|---|---|---|---|
| Fast | 70.9 % | 74.1 % | 76.4 % | 245 of 356 |
| Careful | 76.5 % | 80.0 % | 81.8 % | 188 of 287 |
| Deep | 81.5 % | 86.0 % | 88.6 % | 113 of 226 |
<!-- END loose -->

(The partly covered faces are, by their median height, bigger than the ones no zone touched: a bigger face is more often found, then covered a little short of the 95 % rule.)

## The 30 blind real-estate-kind photos

The 30 blind photos were chosen to look like what an agent takes; only six of them hold labelled faces, and two of those (a dusk crowd on a balcony and a crowd
in a library) hold most of the 78 faces, so these 78 faces are few photos' worth and not independent. Per photo:

<!-- BEGIN blind-photos -->
| Photo | Picture size | Faces | Median face height | Fast hides | Careful hides | Deep hides |
|---|---|---|---|---|---|---|
| blind-004.jpg | 1920x1200 | 2 | 37 px | 1 of 2 | 1 of 2 | 1 of 2 |
| blind-005.jpg | 1920x1016 | 1 | 67 px | 1 of 1 | 1 of 1 | 1 of 1 |
| blind-010.jpg | 1280x960 | 4 | 33 px | 1 of 4 | 1 of 4 | 1 of 4 |
| blind-019.jpg | 1920x1275 | 1 | 39 px | 1 of 1 | 1 of 1 | 1 of 1 |
| blind-028.jpg | 1920x1440 | 44 | 25 px | 24 of 44 | 25 of 44 | 25 of 44 |
| blind-030.jpg | 1920x1280 | 26 | 36 px | 18 of 26 | 18 of 26 | 20 of 26 |
<!-- END blind-photos -->

Each face that Fast, Careful or Deep left unhidden (32 faces) was looked at by eye, enlarged in its photo, and tagged. A face can carry several tags; the "main cause" is the one that
most plausibly kept the detector from it. The tags are judgements, not measurements:

<!-- BEGIN blind-tags -->
| What the face looks like (by eye) | Careful misses: main cause | Careful misses: any cause | Fast misses: any cause | Deep misses: any cause |
|---|---|---|---|---|
| motion blur or out of focus | 1 | 4 | 4 | 3 |
| dark or grainy scene (dusk, shade) | 6 | 19 | 20 | 19 |
| partly hidden by a hand, hair, a hood, another person or a rail | 9 | 12 | 13 | 12 |
| turned: profile, three-quarter, looking down or the back of the head | 12 | 18 | 19 | 16 |
| small or far: under 40 px, upright and in the light, nothing else wrong | 3 | 5 | 5 | 5 |
| **Misses in all** | 31 | 31 | 32 | 29 |
<!-- END blind-tags -->

## Experimental level, not offered: Deep

Deep is not offered in this release, and the README does not describe it. It was part of the sealed read and it is in the log, so its misses are
described here for completeness:

<!-- BEGIN deep -->
Deep hides 81.5 % (996 of 1222) of the independent faces; it leaves 226 misses, 113 of them with no face zone on them. Of those 226, 54.9 % are under 20 px and 86.7 % under 40 px. It hides 90.2 % of the faces at least 40 px tall. Against Careful it hides 74 faces more and 13 fewer, and the faces it gains have a median height of 15 px: its extra effort goes to the smallest faces.
<!-- END deep -->

## Reproduce it

```
export ANONYMIZER_WIDER_DIR=<the WIDER FACE validation subset: val/ and subset.json>
python3 -s -m bench.miss_breakdown log --sealed --out bench/results/miss-breakdown-log.json --wider-split <wider_face_split.zip>
python3 -s -m bench.miss_breakdown tables --log bench/results/miss-breakdown-log.json --out bench/results/miss-breakdown.json   # exit 2 if the totals differ from the sealed read
python3 -s -m bench.miss_breakdown docs                                                                                   # rewrites the generated blocks of this page
```

The WIDER images are test data and are not distributed; the sealed truth labels are in `docs/truth-dataset`.
`tests/test_miss_breakdown.py` checks the log against the sealed totals, the tables against the log, and this page against the tables.
