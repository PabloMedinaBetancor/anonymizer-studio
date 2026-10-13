# Third-party components

The code of the product is MIT. Nothing under GPL, AGPL, CC BY-NC or a "research only" licence is inside the package, with one exception that the table of libraries states exactly: the FFmpeg that the PyAV wheel brings is built with GPL encoders. This file lists
every component that ships with it, its licence and, for each model, what it was trained on. A reading of a licence here is the
author's reading, not legal advice. It is generated from `scripts/third_party.json` by `scripts/third_party.py`; edit the data, not this file.

## Models

| Model | Licence of the code and the weights | Trained on | Note |
|---|---|---|---|
| YuNet 2023mar | MIT (OpenCV Zoo `LICENSE`) | WIDER FACE training set | WIDER FACE's own terms are research-only (CC BY-NC-ND 4.0). The licence of that data is a risk, and it is stated here rather than hidden: these are the weights the product ships. |
| RetinaFace ResNet-34 (yakhyo) | MIT (`yakhyo/retinaface-pytorch` `LICENSE`); weights trained by the repository's author | WIDER FACE training set | Same data note as YuNet. The other candidates measured by the pilot (RetinaFace R18, R50, OMZ face-detection-0205/0206, CenterFace) are not shipped. |
| YOLOX-nano | Apache-2.0 (Megvii-BaseDetection/YOLOX) | COCO 2017 | Vehicles and persons; never blurred, context only. |
| RF-DETR-N (nano), 384x384 (Roboflow) | Apache-2.0 (roboflow/rf-detr, `LICENSE` and the README table, row RF-DETR-N, 384x384, 30.5 M parameters, "Apache 2.0"; read on 03/10/2026, branch `develop`) Weights: https://github.com/ankandrew/open-image-models/releases/download/assets/rf-detr-nano-384-coco.onnx (SHA-256 `fdcdd1c566ed4ab4d3700ca1ce6f03edadfec2a0655432e40e4c2734623feaec`). Licence text: https://github.com/roboflow/rf-detr/blob/develop/LICENSE. "The open-source `rfdetr` package and Apache-designated models are released under Apache 2.0, while Plus components (`rfdetr_plus`, including RF-DETR-XL/2XL detection models) are licensed under PML 1.0." | COCO 2017 (the DINOv2 backbone it starts from is Apache-2.0 and was pre-trained on a curated web-image set, as its authors state) | The default vehicle detector of the plate stage (`Options.vehicle_model="rfdetr"`) and an optional person detector; never blurred, context only. The file is the ONNX export of Roboflow's RF-DETR-N checkpoint published as a release asset of `ankandrew/open-image-models` (that repository is MIT; its release page states no licence of its own for the export, so the weights' licence is Roboflow's). Only the Apache-designated sizes (N, S, M, L) are usable: RF-DETR-XL and 2XL are PML 1.0 and are not this model. YOLOX-nano stays available with `vehicle_model="yolox"`. |
| PP-OCRv6 text detector (RapidOCR's ONNX file) | Apache-2.0 (PaddleOCR; RapidOCR) | PaddleOCR's training mix (public and internal text data; the project does not publish the list in one place) | The text detector of the plate path (the wrapper is `classes/text.py`). |
| PP-OCRv5 Latin text recogniser (RapidOCR's ONNX file) | Apache-2.0 (PaddleOCR; RapidOCR, whose package metadata says Apache-2.0; the same family and the same source as the PP-OCRv6 detector above) Weights: https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx (SHA-256 `b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a`). Licence text: https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE. "Read on 11/10/2026 from three sources: PaddleOCR's own LICENSE (raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE: "Apache License, Version 2.0, January 2004"), the Hugging Face model card PaddlePaddle/latin_PP-OCRv5_mobile_rec (metadata "License: apache-2.0") and RapidOCR's LICENSE (Apache License 2.0, which packages this ONNX file); also the metadata of the RapidOCR 3.9.2 package (read on 06/10/2026)" | PaddleOCR's training mix (public and internal text data; the project does not publish the list in one place) | The plate reader (added in 2.2; the wrapper is `classes/platetext.py`): reads the crop of a text region on a vehicle so that a plate ("4314 KMJ") can be told from a word ("TAXI", "RENAULT"). It never blurs anything alone: it takes a word out of the default blur and brings a plate the position rule rejected into it. |

## Libraries

| Component | Licence | Used for |
|---|---|---|
| ONNX Runtime | MIT | runs the ONNX models on the CPU |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | image work and YuNet |
| NumPy | BSD-3-Clause | arrays |
| PyAV (`av`) | BSD-3-Clause; the wheel of PyAV 17.1.0 bundles an FFmpeg under LGPL v3 or later, built with libx264 and libx265, which are GPL | video reading and writing |
| Pillow | HPND | photo reading and writing |
| pywebview (optional, the Studio window) | BSD-3-Clause | window around the Studio |

## Test and benchmark data (not shipped)

Public photos and clips under CC0, CC BY, CC BY-SA or public domain, listed with their sources in `bench/SOURCES.csv` and `demo/SOURCES.csv`.
The OpenALPR EU plate photos (AGPL-3.0) and WIDER FACE validation images (CC BY-NC-ND 4.0) stay on the test machines and are not
distributed; a measurement published on them is not a copy of them. The per-face logs in `bench/results` hold, for WIDER FACE, only heights, results and attribute codes under opaque ids: no file name, no box, no picture. The v1 licence-plate model (YOLOv9-t 384 from `ankandrew/open-image-models`, trained with the author's fork of the GPL-3.0 `WongKinYiu/yolov9`) was the baseline of the plate comparison and is **not shipped and not cleared** (the author wrote that he is no lawyer on whether the GPL reaches the weights; issues 14, 20, 22 of that repository). Its file is not in `models/SHA256SUMS`. `Options(plate_path="yolo")` and `PlateModule` remain only as a reference path to reproduce that comparison with a file the person fetches themselves (`scripts/fetch_reference_plate_model.py`); the product's plates are found by the text path.
