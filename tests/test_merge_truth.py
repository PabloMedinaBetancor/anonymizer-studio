import unittest

from bench import merge_truth


def entry(*boxes):
    return {
        "width": 100,
        "height": 100,
        "boxes": [{"box": b, "kind": "plate", "readable": True, "ignore": False} for b in boxes],
    }


class MergeTruthTest(unittest.TestCase):
    def test_adds_only_boxes_without_a_partner(self):
        author = {"set": "s", "licence": "mixed", "images": {"a.jpg": entry([10, 10, 30, 20]), "b.jpg": entry()}}
        verifier = {"set": "s", "licence": "mixed", "images": {"a.jpg": entry([11, 10, 31, 21], [50, 50, 70, 60])}}
        merged, added = merge_truth.merge(author, verifier)
        boxes = [b["box"] for b in merged["images"]["a.jpg"]["boxes"]]
        self.assertEqual(boxes, [[10, 10, 30, 20], [50, 50, 70, 60]])
        self.assertEqual(len(added), 1)
        self.assertEqual(author["images"]["a.jpg"]["boxes"], entry([10, 10, 30, 20])["boxes"])

    def test_images_outside_the_list_stay_unchanged(self):
        author = {"set": "s", "licence": "mixed", "images": {"a.jpg": entry()}}
        verifier = {"set": "s", "licence": "mixed", "images": {"a.jpg": entry([5, 5, 20, 15])}}
        merged, added = merge_truth.merge(author, verifier, only={"other.jpg"})
        self.assertEqual(merged["images"]["a.jpg"]["boxes"], [])
        self.assertEqual(added, [])


if __name__ == "__main__":
    unittest.main()
