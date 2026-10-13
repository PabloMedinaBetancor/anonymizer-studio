# Face ground truth

Labelled photos for the face benchmark (`bench/bench_faces.py`). Two sets:

- `set-a.json`: photos with faces (crowds, markets, groups, street, terraces, close-ups, faces from 10 to 400 px).
- `set-b.json`: photos without people, chosen to fool a face detector (statues, paintings, murals, mannequins,
  car grilles, door knockers, clocks, fruit, empty interiors, people seen from behind).

The images are not in git (`$ANONYMIZER_BENCH_DATA/faces/set-a|set-b/`, fetched by `bench/fetch_faces_data.py` from
`bench/faces_titles.csv`). Every file has a row in `bench/SOURCES.csv` with its Commons page, author and licence
(CC0, CC BY or CC BY-SA only). Images are the 1920 px (or smaller) thumbnails Commons serves.

## Format

```
{"set": "a", "licence": "...",
 "images": {"<file name>": {"width": W, "height": H,
                            "boxes": [{"box": [x0, y0, x1, y1], "kind": "face", "ignore": false, "note": ""}]}}}
```

Pixels of the image as stored. `bench/bench_faces.py` refuses to run when width and height do not match the file,
and `tests/test_faces_truth.py` checks the structure of both files.

## What counts

- `ignore: false`: a face of a real person (also in a photo, a screen or a photographic poster) that a reviewer
  looking at the enlarged image can tell is a face. Counted in recall.
- `ignore: true`: neither counted nor penalised. Used for statues, paintings, drawings, graffiti, dolls and
  mannequins (`note` says which), faces cut by the frame, faces behind glass or too blurred to tell, and tiny
  or ambiguous faces in dense crowds (`unclear`, `tiny_crowd_face`), plus whole dense-audience regions where faces
  are under 10 px (`dense_audience_faces_under_10_px`). A blur on an ignored box is not a false blur and not a hit.
  The benchmark reports how many ignored boxes were blurred.

## How it was made

One reviewer (a vision model that looks at images), no second pass yet. For each photo:

1. YuNet ran at a threshold of 0.25 over the whole image, over the image at full size and over 400 px tiles
   enlarged 2x, giving numbered candidate boxes (the box of a labelled face is YuNet's box).
2. The reviewer looked at the photo with the candidates drawn over it (and at enlarged regions when faces were
   small), threw out candidates that are not faces (hats, backs of heads, hands, text, posters) and added by hand
   the faces YuNet did not propose when they were visible (`add`; these include tilted, behind-glass and
   profile faces).
3. In crowds with too many faces to check one by one, a candidate with a YuNet score of at least 0.8 and a side of
   at least 10-25 px (varies per photo, see the notes) was accepted after a visual check of the overview; weaker
   candidates were marked `ignore`.

4. After the first benchmark run the reviewer looked at the crop of every face that no zone covered at all (113 of
   1217 then) and every false blur. 56 of those 113 labels were wrong or not defensible (backs of heads, a head turned
   down, onions, a bag, hands over faces, faces behind glass, out of focus, under 12-14 px) and were turned into
   `ignore` boxes with the reason in `note`; the rest stayed as misses. One "false blur" turned out to be a real face
   that had not been labelled and was added. Mistakes in the other direction (a label that is wrong but was found by
   the detector) were only checked on a random sample: 45 labelled faces drawn at random from set a (seed 1) were
   looked at, 41 were clearly faces and 4 were doubtful (a head down, a dark blob, a back of a head, a face turned
   away), so expect about 5-10 % of the labelled boxes in crowds to be questionable. Both facts push recall up a
   little; the miss list in `faces-details.json` is where to look.

Consequences to keep in mind when reading the numbers:

- Recall is measured against faces that are visible to a reviewer **and** that YuNet can find at a low threshold at
  some scale, plus the faces added by hand. Faces YuNet cannot detect at all, but that were not noticed, are
  missing from the denominator. Read the recall as an optimistic bound.
- Accepted boxes in crowds are mostly confident detections, so the faces the earlier application's rule struggles with (small, weak
  scores) are mostly in the `ignore` boxes. The count of ignored boxes that got blurred is reported next to recall.
- The box is YuNet's (brow to chin, a bit narrower than the head). The check is "a zone covers at least 95 % of the
  box", so margins matter. The ellipse that is really drawn covers a little less than the rectangle; both are
  reported.
- Set b has no counted faces by construction, so `false_blur_per_100` is the rate at which BLUR lands on something
  that is neither a face nor a labelled statue, painting, mural, mask or mannequin.

A second reviewer who has not seen these labels redid a small sample: see the next section.

## Blind check of the truth (VER-F2, 03/10/2026)

A verifier with no detector, no proposals and no labels of the author (a blind verifier) labelled 15 photos of
set a by hand: the sample is `bench/pick_blind_faces_sample.py` (1 to 8 counted faces and at most 8 boxes per photo, seed
20261003, 8 photos that have a miss or a false blur in the bench and 7 at random). Command (reads its inputs, writes
`bench/results/truth-check-faces.json`):

    python3 -s -m bench.truth_check_faces --verifier VERIFIER.json --images verify-faces-blind.txt [--merge]

Boxes are paired at IoU 0.3 (hand boxes differ by 3-10 px; at 0.5 the concordance drops to 72 % / 84 %, it is a box-size
effect, not a labelling one). Result: 30 pairs agree, 11 pairs are `ignore` on a side, 1 box only the verifier has, 5 boxes
only the author has. Every unpaired box was read by eye on its crop (`truth-check-review.json`):

- The one only the verifier has is a real face (a man in profile behind a bus window): **the author missed 1 face of 31**
  the verifier counted (coverage 96.8 %). It is now in set a (union, as for plates; `--merge`, idempotent).
- Of the 5 only the author has: 1 is a real small face (a rider in a keffiyeh, 28 px) the verifier missed, and 4 are faces
  of framed portraits on a wall (the R6 case, ambiguous): they stay as the author labelled them.
- No box of the author was found wrong for being a non-face other than the framed portraits.

Reading: on few-face photos the author's truth missed about one real face in thirty, and the blind labeller missed
about as many; the bias of the YuNet-proposal method is small here. What this does NOT show: it covers 15 photos and
31 faces (the interval is wide), only photos with few faces (crowds, where the author's truth is weakest because most
boxes are `ignore`, were not re-labelled blind), one verifier, and half the photos were picked because the bench fails
on them. An earlier 60-photo verifier truth was not blind (it used YuNet) and is not used.
After the union the bench was rerun on the benchmark machine (i3-6006U, one machine, 128 photos): BLUR 77.71 % (was 77.78 %),
BLUR or CHECK 95.09 % (was 95.18 %), false BLUR 0.0 per 100 photos without people, 0.97 s per photo.

## Video (F4): a small sample, and why no recall figure is published

`video.json` holds hand-made truth for 52 frames of two public clips (fetched by `bench/fetch_faces_video.py`, rows in
`bench/SOURCES.csv`): 28 frames of "Calle Preciados Madrid" (CC BY-SA 3.0, 1280x720, 1,056 frames) and 24 of "Pasejo en
Hong Kong Island (2014)" (CC BY-SA 4.0, 1920x1080, 323 frames), evenly spaced (frame index `round((k + 0.5) * N / n)`).
`bench/bench_faces_video.py` runs the video analysis (Standard and Maximum) and counts, per labelled face, whether the
blur the render would draw covers at least 80 % of the box (and at least 50 %, reported apart).

How it was labelled: one reviewer (a vision model), by eye, **with no detector proposals** (unlike the photos), on
frames enlarged 1.2x to 8x with a coordinate grid. Boxes are hand-drawn, good to about 3 px. A face is counted when
it can be told to be a face (profile faces count); backs of heads, faces under about 10 px in the far crowd and
faces too dark or blurred to tell are not counted (`ignore` boxes mark the doubtful ones near a counted face; the many
tiny far faces are simply not labelled).

What came out, on the benchmark machine (i3-6006U, `hardware` in `bench/results/faces-video.json`):

| | Standard | Maximum |
|---|---|---|
| Preciados, faces fully blurred (>= 80 % covered) | 9 of 22 (41 %, 95 % interval 23-61) | 7 of 22 (32 %, 16-53); 8 of 22 in the first run of the same code |
| Preciados, faces at least half covered | 17 of 22 (77 %, 57-90) | 15 of 22 (68 %, 47-84) |
| Hong Kong, tracks blurred in 323 frames | 0 | 0 |
| analysis time per frame | 74 ms (720p), 63 ms (1080p) | 345 ms (720p), 417 ms (1080p) |

The two runs of Standard gave the same faces; Maximum gave 8 and then 7 on the same machine, the same code and the same truth
(the times also moved, 94 and 74 ms per frame for Standard). Nothing in the sample can tell which is the "real" figure, which is one
more reason not to quote it.

Why these are **not** published as the video recall of the product:

- 22 faces from one street clip, one reviewer and no second pass: the interval is 23-61 %, and Standard against
  Maximum (9 against 8 faces) is inside the noise. This sample cannot say whether Maximum finds more faces.
- Almost all counted faces are profiles or turned heads in deep shadow, at 25-45 px: the hardest case, not the average
  one. The clip has no face looking at the camera in the sharp light, so it says nothing about walk-throughs of homes.
- The 80 % cover rule is hard on an ellipse (an ellipse covers about 78 % of its own rectangle) and on hand boxes
  good to 3 px, which is why 8 more faces are "at least half covered": they are not leaks of the whole face, but they
  are not blurred either. Neither threshold is "right".
- Hong Kong has no face in its frames (a woman walking away down a corridor): it measures false blurs only, 0 in the
  whole clip at both qualities, and gives no recall.
- Frames where the analysis touched nothing: Preciados 245 and 320 (both qualities), 169 (Maximum). Those are the real
  misses; the rest are partial.

What would make a figure defendable: a second reviewer who has not seen these boxes labelling the same 28 frames, more
clips with faces looking at the camera (Commons has few CC videos with people; none was found that is not a crowd
of tiny faces), and a count of at least 100 faces.
