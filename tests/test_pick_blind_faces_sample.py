"""bench/pick_blind_faces_sample.py: few faces, fixed seed, half from the flagged photos, nothing but names."""

import json
import unittest

from bench import pick_blind_faces_sample as pick


class PickBlindFacesSampleTests(unittest.TestCase):
    def test_sample_is_small_deterministic_and_few_faces(self) -> None:
        a = pick.pick(15, 20261003)
        self.assertEqual(a, pick.pick(15, 20261003))
        self.assertEqual(len(a), 15)
        with open(pick.TRUTH_A, encoding="utf-8") as f:
            images = json.load(f)["images"]
        with open(pick.DETAILS, encoding="utf-8") as f:
            d = json.load(f)
        flagged = {x["image"] for x in d["misses"] + d["false_blurs"]}
        for line in a:
            self.assertTrue(line.startswith("set-a/"))
            boxes = images[line[6:]]["boxes"]
            faces = sum(1 for x in boxes if not x["ignore"])
            self.assertTrue(1 <= faces <= pick.MAX_FACES)
            self.assertLessEqual(len(boxes), pick.MAX_BOXES)
        self.assertGreaterEqual(sum(1 for line in a if line in flagged), 8)


if __name__ == "__main__":
    unittest.main()
