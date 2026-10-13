"""Shared bits of the face tests: skip when something outside git is missing, unless ANONYMIZER_REQUIRE_REFERENCE=1
says it must be there."""

from __future__ import annotations

import os
import unittest
from typing import NoReturn

from anonymizer import contract as c
from anonymizer.classes.faces import FaceModule


def missing(what: str) -> NoReturn:
    if os.environ.get("ANONYMIZER_REQUIRE_REFERENCE"):
        raise AssertionError(f"{what} is required here and was not found")
    raise unittest.SkipTest(f"{what} not found")


def make_module() -> FaceModule:
    try:
        return FaceModule()
    except c.ModelLoadError as e:
        missing(f"face model ({e.detail})")
