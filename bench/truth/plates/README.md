# Plate ground truth

One JSON file per image set, in this folder; the images themselves live outside git, in
`$ANONYMIZER_BENCH_DATA/plates/<set>/` (see `bench/SOURCES.csv` for where each one came from and under which licence).

```json
{"set": "<name>", "licence": "<SPDX-like id>",
 "images": {"<file>": {"width": 2048, "height": 1536,
                       "boxes": [{"box": [x0, y0, x1, y1], "kind": "plate", "readable": true, "ignore": false}]}}}
```

- `box` is in pixels of the image as `cv2.imread` returns it (EXIF orientation applied), `0 <= x0 < x1 <= width`.
- `readable`: a person could read the characters at the size they have in the image.
- `ignore`: the box is unclear (a sign that might be a plate, a plate cut in half, a reflection). The benchmark
  counts neither a hit nor a miss on it, and a blur over it is not a false blur.
- A plate that is visible and not marked `ignore` is a plate the benchmark expects to see blurred. Plates of
  vehicles that are only partly in the frame are marked when their characters are still partly visible.

`tests/test_plates_truth.py` checks every file against this schema (`bench/plates_truth.py: problems`).

## Sets

### openalpr-eu (108 photos, AGPL-3.0, local benchmark only)

Converted by `bench/convert_openalpr_eu.py` from the `.txt` files of `endtoend/eu` in
github.com/openalpr/benchmarks: each one holds the file name, `x y width height` and the plate text; only the box is
kept here (the text is not copied). Every plate is `readable: true`, as annotated upstream. A photo can show other
plates that upstream did not annotate (the 22 px plate of `eu7.jpg` is one), so false blurs are not measured on
this set. Images stay out of git and out of the demo.

`openalpr-eu.json` follows the licence of the upstream annotations, AGPL-3.0. The licence text is `AGPL-3.0.txt` in this folder,
and `docs/truth-dataset/labels-agpl/LICENSE` carries the same text next to the copy of the labels in the dataset.

### streets (Wikimedia Commons photos of streets, car parks and facades; CC0, CC BY, CC BY-SA)

Fetched by `bench/fetch_commons_streets.py` (one request per second, 1920 px wide; each file has its page, author
and licence in `bench/SOURCES.csv`; the set's `licence` is `mixed` and the real one is per file there). Labelled by
looking at every photo at full size with the detector's candidates drawn on it and a coordinate grid
(`bench/label_streets.py`), and at enlarged crops of every candidate and of every spot where a plate could be:
1. Every plate whose shape is clearly a plate gets a tight box, `readable` when the characters can be made out. A
   plate seen only as a few pixels, or edge on, is `readable: false` but still expected to be blurred.
2. A box that might be a plate but cannot be told (very dark, tiny, behind a bar) is `ignore: true`: neither a hit
   nor a miss, and a blur over it is not a false blur.
3. The detector's candidates were only a guide. They were not copied: where a candidate was a headlight, a window, a
   sticker or a sign it was left out, and plates the detector did not propose were added by eye (street_031 and
   street_032, both small and angled). Photos with distant cars whose plates cannot be seen at all (street_002,
   _026, _036) have no boxes. In street_060 (a car park with hundreds of cars) only the plates the detector
   proposed and that could be confirmed are labelled; the far rows are not, so that photo can only produce
   false-blur counts for the plates nearest the camera.
4. About half of the photos have no vehicle at all (street name signs, facades, windows, shop fronts, a garage
   door) because they are what produces false blurs.

### streets: blind verification (03/10/2026)

A second reviewer that had not seen this truth labelled 30 of the 60 street photos (every photo with a miss or a false
blur in the first benchmark run, plus a seeded random pick; list by `bench/pick_verify_sample.py`). The files are
`streets-author.json` (the original), `streets-verifier.json` (the blind one) and `streets.json`, which the benchmark
reads: the author's boxes plus the verifier's boxes that had no partner at IoU >= 0.3 (`bench/merge_truth.py`; 11 boxes
added, looked at one by one in enlarged crops: six small plates on parked motorbikes in street_027, a plate on a bus in
street_005, and unclear ones marked `ignore`). Agreement before merging, from `bench/compare_truth.py` (report in
`bench/results/truth-check-plates.json`): the two labellers disagreed mostly on tiny plates in dense car parks and on
parked motorbikes; the plates they both saw they boxed alike. Numbers measured against the merged truth are the ones
published; the 30 photos outside the sample were not double-checked.
