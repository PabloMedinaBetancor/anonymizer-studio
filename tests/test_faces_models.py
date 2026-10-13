"""Heavy face tests: they load YuNet and run it on public photos (flat memory, F6; small faces, R6; the 1.04 cases are ported here). Off unless ANONYMIZER_HEAVY=1 and, on a small machine, run with a memory cap and nothing else heavy running:

    ANONYMIZER_HEAVY=1 python3 -s -m unittest tests.test_faces_models -v

With the switch on, a missing model or photo fails the test instead of skipping it when
ANONYMIZER_REQUIRE_REFERENCE=1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import unittest

import cv2
import numpy as np

from anonymizer import contract as c
from tests.faces_support import make_module, missing

HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
SKIP_REASON = "set ANONYMIZER_HEAVY=1 to run the tests that load the models"
DATA = os.environ.get("ANONYMIZER_BENCH_DATA", "")
PHOTOS = os.path.join(DATA, "faces", "set-a")
TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "truth", "faces", "set-a.json")
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "bench_faces.py")

# a distant group of tourists (13 faces of 15-30 px) stands in for the property photo of the 1.04 case R6 (an 18 px face in a
# framed wedding photo); the framed portraits of a 1940s room are a harder case that is not solved yet
DISTANT_GROUP = "Tourists-taking-pictures-at-Prime-Meridian-monument-Greenwich-Observat.jpg"
FRAMED_PORTRAITS = "Woman-holding-dog-in-living-room-AM-79400-1.jpg"


def photo_names(count: int) -> list:
    if not os.path.isdir(PHOTOS):
        missing(f"benchmark photos in {PHOTOS}")
    names = sorted(n for n in os.listdir(PHOTOS) if n.endswith(".jpg"))
    if len(names) < count:
        missing("enough benchmark photos")
    step = len(names) // count
    return names[::step][:count]


def read(name: str) -> np.ndarray:
    return cv2.imread(os.path.join(PHOTOS, name))


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


@unittest.skipUnless(HEAVY, SKIP_REASON)
class MemoryAndSmallFaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = make_module()

    def test_F6_memory_stays_flat_over_2000_retries_of_varied_sizes(self):
        frames = [read(n) for n in photo_names(4)]
        rnd = random.Random(11)

        def one_retry():
            bgr = rnd.choice(frames)
            height, width = bgr.shape[:2]
            side = rnd.uniform(9, 260)
            x, y = rnd.uniform(0, max(1, width - side)), rnd.uniform(0, max(1, height - side))
            self.module.retry(bgr, (x, y, x + side, y + side), rnd.choice(list(c.RetryPurpose)))

        for _ in range(300):
            one_retry()
        before = rss_mb()
        for _ in range(2000):
            one_retry()
        self.assertLess(rss_mb() - before, 25.0)

    def test_F1_retries_never_return_a_score_under_the_network_cut(self):
        # 1.04 asked for 0.3 and 0.35 but builds YuNet at 0.5, so nothing below 0.5 ever comes back
        bgr = read(photo_names(1)[0])
        rnd = random.Random(5)
        height, width = bgr.shape[:2]
        for _ in range(300):
            side = rnd.uniform(10, 200)
            x, y = rnd.uniform(0, width - side), rnd.uniform(0, height - side)
            found = self.module.retry(bgr, (x, y, x + side, y + side), c.RetryPurpose.VERIFY)
            if found is not None:
                self.assertGreaterEqual(found.score, 0.5)

    def small_faces_not_covered(self, name):
        """Labelled faces of 15-30 px of `name` that no BLUR or CHECK zone covers, and how many there are."""
        if not os.path.isfile(TRUTH):
            missing("bench/truth/faces/set-a.json")
        with open(TRUTH, encoding="utf-8") as f:
            labelled = json.load(f)["images"][name]["boxes"]
        spec = importlib.util.spec_from_file_location("bench_faces", BENCH)
        bench = importlib.util.module_from_spec(spec)
        sys.modules["bench_faces"] = bench
        spec.loader.exec_module(bench)
        zones = bench.decide_photo(self.module, read(name))
        small = [
            b["box"]
            for b in labelled
            if not b["ignore"] and 15 <= max(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]) <= 30
        ]
        return [box for box in small if bench.coverage(box, zones, False) < bench.COVERED], len(small)

    def test_R6_small_faces_of_a_distant_group_come_out_blur_or_check(self):
        uncovered, total = self.small_faces_not_covered(DISTANT_GROUP)
        self.assertGreaterEqual(total, 10)
        self.assertEqual(uncovered, [])

    @unittest.expectedFailure
    def test_R6_known_gap_framed_portraits_in_a_dark_room_are_not_found(self):
        uncovered, total = self.small_faces_not_covered(FRAMED_PORTRAITS)
        self.assertGreaterEqual(total, 4)
        self.assertEqual(uncovered, [])


if __name__ == "__main__":
    unittest.main()
