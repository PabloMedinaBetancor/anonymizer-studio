# Anonymizer truth labels

Boxes of faces and licence plates on public photos and clips, and how they were made. **Labels only: no image or video is in this
package.** Every image and clip comes from Wikimedia Commons (or, for one set, the OpenALPR benchmark) and has its page, licence, author
and the SHA-256 of the exact file the boxes were drawn on in `IMAGES.csv`; fetch the file from its page and compare the hash before you use the labels.

## What is in it

| File | What |
|---|---|
| `labels/faces/*.json`, `labels/plates/*.json` | the boxes, one file per set, in pixels of the image as stored; format below |
| `labels-agpl/plates/openalpr-eu.json` | plate boxes converted from the annotations of the OpenALPR benchmark (AGPL-3.0); kept apart because of that licence |
| `split/split-v2.json` | which photo is development and which is sealed, by the SHA-256 of the file (not its name), with the date the blind photos were added: the sealed part is the one the v2 results of the product were read on, once; the sealed labels are in this package |
| `MANIFEST.csv` | every label file with its size, SHA-256 and the licence of its labels |
| `IMAGES.csv` | every labelled image or clip: set, name, size, number of boxes and of `ignore` boxes, source page, licence of the image, its author (as the source gives it), SHA-256 of the file |

`python3 scripts/make_truth_dataset.py` rebuilds the package from `bench/truth` and `bench/SOURCES.csv` and `--check` tells if it is out of date.

## Format

```
{"images": {"<file>": {"width": W, "height": H,
                       "boxes": [{"box": [x0, y0, x1, y1], "kind": "face" | "plate", "ignore": false, "note": ""}]}}}
```

Video files have `{"clips": {"<id>": {"file": "...", "frames": {"<frame number>": {"boxes": [...]}}}}}` instead.
`ignore: true` means the box is not counted either way: it is a statue, a poster, a face behind glass or too dark to judge, a plate that
cannot be told from a sign. A plate has `readable` (a person could read the characters at the size they have) and photos of vehicles carry
the vehicle boxes the plate stage was measured with.

## Method

- **Plates, streets.** Photos of streets, car parks and facades looked at one by one at full size with the detector's candidates and a
  coordinate grid, then at enlarged crops. Every plate that is clearly a plate has a tight box. The candidates were a guide, not a copy:
  headlights, signs and stickers were left out and plates nobody proposed were added by eye. A second labeller who had not seen the first
  truth redid half of them (`streets-author.json`, `streets-verifier.json`); `streets.json` is the union the benchmark reads.
- **Plates, v2.** A newer set labelled blind by hand, with no detector proposals: 108 photos, 159 plates to count and 21 `ignore` boxes (`plates-v2.json`); the vehicles are boxed on photos 001 to 040 only. 74 photos were the development part and 34 the sealed part (see `split/split-v2.json` below).
- **Faces, photos.** Built from detector proposals at a low threshold, judged one by one, then checked blind on a sample. Faces that no
  detector proposed and the reviewer did not notice are missing, so this truth is an optimistic bound. `blind-faces-photos.json` was labelled by
  hand by a labeller with no proposals.
- **Faces, video.** Evenly spaced frames of public clips, from proposals judged on crops; `blind-video-frames.json` is the part labelled with no proposals. Its 12 frames have no row of their own in `IMAGES.csv` (their source columns are empty): they are frames of the clips listed for `truth-v2-video.json`, and the label file records the hash of its source.
- **Plates, OpenALPR EU.** Converted by `bench/convert_openalpr_eu.py` from the upstream annotations: the box is kept, the plate text is not.

## Licences

- The boxes this project drew: CC BY 4.0.
- `labels-agpl/`: boxes derived from the OpenALPR annotations, so under AGPL-3.0. The licence text is `labels-agpl/LICENSE`.
- The images and clips are not distributed. Each keeps the licence in `IMAGES.csv` (CC0, CC BY, CC BY-SA, public domain, or AGPL-3.0 for the OpenALPR photos).
