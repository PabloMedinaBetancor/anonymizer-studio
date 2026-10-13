# Models

Every model the product loads. `models/SHA256SUMS` holds the SHA-256 of each file; the product checks it at load and a file that
is missing, unlisted or changed is a `ModelLoadError` before any photo is touched. Licence notes, training data and the components
that are not models are in `THIRD-PARTY.md`.

| File | What | Licence | Source | SHA-256 |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (OpenCV Zoo): the light face model of the first pass, and the same detector run again on an enlarged zone as a cross-check | MIT | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | see `models/SHA256SUMS` |
| `retinaface_r34.onnx` | RetinaFace ResNet-34 (yakhyo/retinaface-pytorch, release v0.0.1), dynamic input: the heavy face model of the second look | MIT | https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx | see `models/SHA256SUMS` |
| `yolox_nano.onnx` | YOLOX-nano COCO: vehicles (car, motorcycle, bus, truck) for the plate path (opt-in `vehicle_model="yolox"` since RF-DETR is the default) and persons (class 0) as context for the second look | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX | see `models/SHA256SUMS` |
| `rf-detr-nano-384-coco.onnx` | RF-DETR-N (nano) COCO, 384x384 (Roboflow; ONNX export of the open-image-models release): the default vehicle detector of the plate stage (`Options.vehicle_model="rfdetr"`, `classes.load(..., vehicle_model="rfdetr")`) and optional person detector of the second look (`PersonFinder(person_model="rfdetr")`); about ten times the time of YOLOX-nano, finds the small vehicles it misses. In the default install | Apache-2.0 (roboflow/rf-detr, RF-DETR-N; XL and 2XL are PML 1.0 and are not this model; text and quote in `THIRD-PARTY.md`) | https://github.com/ankandrew/open-image-models/releases/download/assets/rf-detr-nano-384-coco.onnx | see `models/SHA256SUMS` |
| `PP-OCRv6_det_small.onnx` | PP-OCRv6 text detector, ONNX build of RapidOCR (the text path of plates) | Apache-2.0 | https://github.com/RapidAI/RapidOCR (model of PaddleOCR, https://github.com/PaddlePaddle/PaddleOCR) | see `models/SHA256SUMS` |
| `latin_PP-OCRv5_rec_mobile.onnx` | PP-OCRv5 Latin text recogniser, ONNX build of RapidOCR (added in 2.2): the plate reader, a second opinion on the text regions of a vehicle: it reads the crop so that a plate ("4314 KMJ") is told from a word ("TAXI", "RENAULT"); takes a word out of the default blur and brings in a plate that the position rule rejected, never blurs alone (`classes/platetext.py`) | Apache-2.0 (PaddleOCR; RapidOCR) | https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx | see `models/SHA256SUMS` |

The `.onnx` files are not committed. They sit in `models/` of the product folder (or where `ANONYMIZER_MODELS` or `--models` says) and
are checked against `models/SHA256SUMS` when loaded.

**Not shipped: the v1 plate model.** `yolo-v9-t-384-license-plates-end2end.onnx` (YOLOv9-t 384, `ankandrew/open-image-models`) left the product when the text path took its place: its licence cannot be cleared (THIRD-PARTY.md). It is not in `models/SHA256SUMS`. The reference path `Options(plate_path="yolo")` and the paired comparison of the two plate paths need it: `python3 scripts/fetch_reference_plate_model.py` downloads it into `models/` and checks its SHA-256; it is git-ignored, and while it sits in `models/` the licences gate reports it as an unlisted file on purpose: remove it before the gates run or before a copy is handed over.
