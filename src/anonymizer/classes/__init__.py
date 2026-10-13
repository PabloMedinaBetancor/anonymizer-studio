"""Per-class modules: faces (anonymizer.classes.faces) and plates + vehicles
(anonymizer.classes.plates). The engine only talks to them through contract.ClassModule."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence

from ..contract import DETECTION_MODES, FACE, FIT_MODES, PLATE, ClassModule

PLATE_PATHS = ("yolo", "text")
VEHICLE_MODELS = ("yolox", "rfdetr")


def load(
    kinds: Sequence[str],
    threads: int = 0,
    models: Optional[str] = None,
    plate_path: str = "text",
    detection: str = "fast",
    vehicle_model: str = "rfdetr",
    plate_band: Optional[bool] = None,
    fit: str = "tight",
) -> List[ClassModule]:
    """Build the modules for `kinds` (FACE, PLATE), in that order. Raises contract.ModelLoadError. `plate_path` picks how plates are
    found: "text" (the default: the text detector on vehicles, the PT4 configuration) or "yolo" (the YOLOv9-t plate model of v1,
    a reference path: the weights are not shipped, see MODELS.md). `vehicle_model` picks the vehicle detector of the plate stage: "rfdetr" (the default:
    RF-DETR nano, PT1b, finds the small vehicles YOLOX-nano misses, about ten times slower per picture) or "yolox" (YOLOX-nano). `detection` ("fast", "careful" or "deep", as Options.detection): for the last two the
    face module is the one with the heavy model, and a missing heavy model file is a ModelLoadError that names it, never a silent
    downgrade to the light model (SL7); "fast" never loads a heavy model. `plate_band` (text path, photos only) adds a
    plate-position band on small vehicles (PT4 lever, TextPathThresholds.band; vehicles under 400 px wide, no suppression by text). None,
    the default, means OFF since 2.2 (the band hid whole walls and car sides: decision of 06/10/2026); True asks for it (it needs the text path),
    which is what 2.1's default did in "careful" and "deep". `fit` is "tight" (2.2: the photo margins of the modules are small) or "loose"
    (2.1's margins)."""
    if plate_path not in PLATE_PATHS:
        raise ValueError(f"plate_path must be one of {PLATE_PATHS}, not {plate_path!r}")
    if detection not in DETECTION_MODES:
        raise ValueError(f"detection must be one of {DETECTION_MODES}, not {detection!r}")
    if vehicle_model not in VEHICLE_MODELS:
        raise ValueError(f"vehicle_model must be one of {VEHICLE_MODELS}, not {vehicle_model!r}")
    if plate_band and plate_path != "text":
        raise ValueError("plate_band needs plate_path='text'")
    if fit not in FIT_MODES:
        raise ValueError(f"fit must be one of {FIT_MODES}, not {fit!r}")
    if plate_band is None:
        plate_band = False
    modules: List[ClassModule] = []
    for kind in kinds:
        if kind == FACE:
            if detection == "fast":
                from .faces import FaceModule

                modules.append(FaceModule(threads=threads, models=models, fit=fit))
            else:
                from .heavy import HeavyFaceModule

                modules.append(HeavyFaceModule(threads=threads, models=models, fit=fit))
        elif kind == PLATE:
            from .plates import DEFAULT_TEXT_PATH, PlateModule, TextPlateModule

            module = TextPlateModule if plate_path == "text" else PlateModule
            vehicles = None
            if vehicle_model == "rfdetr":
                from .rfdetr import load_vehicle_detector

                vehicles = load_vehicle_detector(threads, models)  # checked now, opened at the first picture
            extra = {"text_path": replace(DEFAULT_TEXT_PATH, band=True)} if plate_band else {}
            if module is TextPlateModule:
                from .platetext import PlateReader

                extra["fit"] = fit
                extra["reader"] = PlateReader(threads, models)  # checked now, opened at the first region it reads
            modules.append(module(threads=threads, models=models, vehicles=vehicles, **extra))
        else:
            raise ValueError(f"unknown kind: {kind}")
    return modules
