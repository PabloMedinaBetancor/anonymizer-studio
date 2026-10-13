"""E7: blur styles and strength. Ported from the 1.04 blur tests (D1-D6), plus the style and strength
limits of the spec."""

from __future__ import annotations

import time
import unittest

import numpy as np

try:
    import cv2

    from anonymizer.engine import blur as b
except ImportError as e:  # numpy and cv2 are needed
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

PLATE_BOX = (14, 70, 470, 175)


def text_plate() -> np.ndarray:
    img = np.full((240, 520, 3), 235, np.uint8)
    cv2.putText(img, "1234 BCD", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (10, 10, 10), 7, cv2.LINE_AA)
    return img


def plate_of_width(w: int, text_scale: float):
    """A text plate `w` px wide (aspect 4.5:1) with 40 px of background around it, and its box."""
    h = max(7, int(round(w / 4.5)))
    img = np.full((h + 80, w + 80, 3), 235, np.uint8)
    org = (40 + max(1, w // 30), 40 + int(h * 0.8))
    scale, thickness = w / 300 * text_scale, max(1, round(w / 75))
    cv2.putText(img, "1234 BCD", org, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness, cv2.LINE_AA)
    return img, (40, 40, 40 + w, 40 + h)


def with_photo_margin(box, mx=0.10, my=0.18):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (x0 - w * mx, y0 - h * my, x1 + w * mx, y1 + h * my)


class BlurHidesText(unittest.TestCase):
    def test_E7_characters_cannot_be_recovered_in_any_style(self):
        for style in b.STYLES:
            for strength in (b.MIN_STRENGTH, b.DEFAULT_STRENGTH, b.MAX_STRENGTH):
                img = text_plate()
                before = b.residual(img, PLATE_BOX)
                b.blur(img, PLATE_BOX, b.RECT, style, strength)
                after = b.residual(img, PLATE_BOX)
                self.assertGreater(before, 10)
                self.assertLess(after, 1.5, f"{style} at {strength}")

    def test_E7_no_shape_of_the_letters_survives_inside(self):
        # A mask without an opaque core would leave 20-30 % of the original: stretched, the letters read again.
        x0, y0, x1, y1 = PLATE_BOX
        for style in b.STYLES:
            img = text_plate()
            original = img.copy()
            b.blur(img, PLATE_BOX, b.RECT, style)
            inside = img[y0 + 12 : y1 - 12, x0 + 12 : x1 - 12].astype(np.float32).ravel()
            letters = original[y0 + 12 : y1 - 12, x0 + 12 : x1 - 12].astype(np.float32).ravel()
            if inside.std() > 0:
                self.assertLess(abs(float(np.corrcoef(inside, letters)[0, 1])), 0.3, style)

    def test_E7_plates_of_30_px_and_more_pass_the_residual_test_at_the_floor(self):
        for style in b.STYLES:
            for text_scale in (1.2, 1.5, 1.7):
                for w in (30, 34, 40, 50, 60, 75, 90, 135, 200, 300, 450, 700):
                    img, box = plate_of_width(w, text_scale)
                    b.blur(img, with_photo_margin(box), b.RECT, style, b.MIN_STRENGTH)
                    self.assertLess(b.residual(img, box), 1.5, f"{style}, {w} px, text {text_scale}")

    def test_E7_strength_outside_the_slider_is_refused(self):
        img = text_plate()
        for strength in (b.MIN_STRENGTH - 0.1, b.MAX_STRENGTH + 0.1):
            with self.assertRaises(ValueError):
                b.blur(img, PLATE_BOX, b.RECT, b.GAUSSIAN, strength)
        with self.assertRaises(ValueError):
            b.blur(img, PLATE_BOX, b.RECT, "swirl")


class BlurEdges(unittest.TestCase):
    def test_E7_pixels_away_from_the_patch_never_change(self):
        for style in b.STYLES:
            img = text_plate()
            original = img.copy()
            b.blur(img, PLATE_BOX, b.RECT, style)
            far = np.ones(img.shape[:2], bool)
            far[max(0, PLATE_BOX[1] - 60) : PLATE_BOX[3] + 60, max(0, PLATE_BOX[0] - 60) : PLATE_BOX[2] + 60] = False
            np.testing.assert_array_equal(img[far], original[far])

    def test_E7_video_planes_and_face_ellipses(self):
        y = np.random.default_rng(1).integers(0, 255, (300, 300), dtype=np.uint8)
        u = y[::2, ::2].copy()
        b.blur(y, (100, 100, 200, 220), b.ELLIPSE)
        b.blur(u, (50, 50, 100, 110), b.ELLIPSE)
        self.assertLess(b.residual(y, (110, 115, 190, 205)), 1.5)
        self.assertLess(b.residual(u, (55, 57, 95, 103)), 1.5)

    def test_E7_sixteen_bit_planes_keep_their_depth(self):
        plane = (np.random.default_rng(4).integers(0, 65535, (200, 200))).astype(np.uint16)
        b.blur(plane, (40, 40, 160, 160))
        self.assertEqual(plane.dtype, np.uint16)
        self.assertLess(b.residual(plane, (40, 40, 160, 160)), 1.5)

    def test_E7_boxes_on_the_edge_tiny_or_empty_do_not_fail(self):
        z = np.zeros((100, 100, 3), np.uint8)
        for box in [(-20, -20, 30, 30), (90, 90, 150, 150), (10, 10, 11, 11), (50, 50, 50, 80), (0, 0, 100, 100)]:
            for style in b.STYLES:
                b.blur(z, box, b.RECT, style)
                b.blur(z, box, b.ELLIPSE, style)

    def test_E7_a_1600_px_box_takes_under_a_second(self):
        # The gaussian at full size grew with the box: minutes per frame for big faces in early versions.
        big = np.random.default_rng(2).integers(0, 255, (1920, 1920), dtype=np.uint8)
        start = time.process_time()  # CPU time: the load of other processes is not this code's cost
        b.blur(big, (100, 100, 1700, 1500), b.ELLIPSE)
        self.assertLess(time.process_time() - start, 1.0)


if __name__ == "__main__":
    unittest.main()
