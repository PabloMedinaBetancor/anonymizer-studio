# Licence and credits

The models, libraries and sample footage that Anonymizer Studio uses, each with its licence and what the licence asks of you. Read it before you pass the product, its samples or its documentation to someone else. The licence readings here are ours, not legal advice.

## The code

The code of Anonymizer Studio (`src/`, `scripts/`, `tests/`, `bench/`, `examples/`, `packaging/` and the documentation text) is released under the MIT licence (`LICENSE`, Copyright (c) 2026 Pablo Medina). Three things in the repository are not MIT:

- the sample clips and photos in `demo/` and `examples/data/`, and every screenshot, frame or video made from them (`docs/media/demo.mp4`, `docs/media/demo.jpg`, `docs/media/hero.webp`, `docs/media/hero.gif`, `docs/media/hero-still.png`, `docs/screens/`, `docs/assets/first-video-*.png`, `docs/assets/compare-side-by-side.jpg`), under the CC BY-SA licences in the table below;
- the truth labels in `docs/truth-dataset/labels/`, CC BY 4.0, and the plate labels in `docs/truth-dataset/labels-agpl/` and `bench/truth/plates/openalpr-eu.json`, AGPL-3.0 (`MANIFEST.csv` gives the licence of each file in `docs/truth-dataset/`; the label files in `bench/truth/` carry the licence given there for the same data);
- the models, which are not in the repository and keep the licences listed below.

## Models

Six models ship with the product as files in the models folder, not in the package. `scripts/fetch_models.py` downloads them, and the product checks each against its SHA-256 when it loads it. `MODELS.md` has the full table with the checksums, and `THIRD-PARTY.md` the licence texts and what each model was trained on.

| Model | Licence | What it does | Source |
|---|---|---|---|
| YuNet 2023mar | MIT | The light face model | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| RetinaFace ResNet-34 | MIT | The heavy face model | [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) |
| YOLOX-nano | Apache-2.0 | Vehicles and persons, as context | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) |
| RF-DETR nano | Apache-2.0 | The vehicle detector of the plate path | [RF-DETR](https://github.com/roboflow/rf-detr) |
| PP-OCRv6 text detector | Apache-2.0 | Finds text on vehicles | [RapidOCR](https://github.com/RapidAI/RapidOCR) build of a [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) model |
| PP-OCRv5 Latin recogniser | Apache-2.0 | Reads that text | RapidOCR build of a PaddleOCR model |

Apache-2.0 asks you to keep the licence text and notices when you redistribute the files; MIT asks you to keep the copyright notice. YuNet and RetinaFace were both trained on the WIDER FACE training set, whose own terms are for research only. The licence of the models is not in question; the data licence is a risk we state.

A plate model of an earlier version (YOLOv9-t) is not shipped, because it was trained with a GPL-3.0 project and its licence could not be cleared. `plate_path="yolo"` remains as a reference path only, with a file you fetch yourself with `scripts/fetch_reference_plate_model.py`.

## Libraries

ONNX Runtime (MIT) runs the models, OpenCV (`opencv-python-headless`, Apache-2.0) does the image work, NumPy (BSD-3-Clause) the arrays, PyAV (`av`, BSD-3-Clause) the video and Pillow (HPND) the photos; pywebview (BSD-3-Clause) gives the optional window. The wheel of PyAV brings an FFmpeg built with libx264 and libx265, which are GPL, and FFmpeg itself is LGPL version 3 or later. This is the one place where GPL code comes with the product, and the videos the Studio writes are encoded with it. If you redistribute the product, read the terms of PyAV's wheel for your case.

## The footage of the samples

The sample files in `demo/` and `examples/data/` are public footage from Wikimedia Commons, shortened, scaled down and re-encoded for this product; `demo/SOURCES.csv` and `examples/data/SOURCES.csv` give each file's address, SHA-256 and what was done to it.

| File | What it shows | Author | Licence | Original |
|---|---|---|---|---|
| `demo/street-madrid.mp4` | A 15-second clip of a pedestrian street in Madrid | OsvaldoGago | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Calle Preciados Madrid](https://commons.wikimedia.org/wiki/File:Calle_Preciados_Madrid.ogv) |
| `examples/data/street-madrid-5s.mp4` | The first 5 seconds of `demo/street-madrid.mp4`, re-encoded | OsvaldoGago | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | the same file |
| `demo/street-hongkong.mp4` | A 10-second street clip from Hong Kong with nothing to blur | RG72 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Pasejo en Hong Kong Island (2014)](https://commons.wikimedia.org/wiki/File:Pasejo_en_Hong_Kong_Island_(2014).webm) |
| `demo/street-taipei.jpg` | A photo of dense traffic in Taipei | Syced | [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) | [2005 car traffic in Taipei](https://commons.wikimedia.org/wiki/File:2005_car_traffic_in_Taipei.jpg) |
| `demo/street-westminster.jpg` | A photo of a pedestrian crossing in London | Lewis Clarke | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) | [London, Westminster, Bridge Street Pedestrian Crossing](https://commons.wikimedia.org/wiki/File:London_%2C_Westminster_-_Bridge_Street_Pedestrian_Crossing_-_geograph.org.uk_-_1739851.jpg) |

CC BY-SA means that the footage, and anything you make from it, stays under the same licence, and that you credit the author and say what you changed. If you publish a screenshot, a frame or a clip made with one of these files, credit the author as above and put the result under the same licence.

## The data of the measurements

The accuracy figures were measured on public photos and clips listed with their authors in `bench/SOURCES.csv`. The OpenALPR EU plate photos (AGPL-3.0) and the WIDER FACE validation images (research only) are not redistributed; they stayed on the test machines. The labels, boxes only, are in `docs/truth-dataset/`.

## Support

Questions and bug reports: open an issue on GitHub. Work requests go through the author's Upwork profile.

## Credits

Version 2.2.0. Written by Pablo Medina and built with a team of AI agents he directed.
