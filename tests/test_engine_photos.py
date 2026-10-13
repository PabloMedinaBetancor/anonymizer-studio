"""E4: photos in and out, and the decision loop. Ported from the 1.04 tests (gps G1-G5, photos F1-F5, A1, A2, A7-A9), with fixed fake detectors."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

import numpy as np

try:
    import cv2
    from PIL import Image

    from anonymizer.engine import photos as ph
    from anonymizer.engine.blur import residual
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from anonymizer.contract import FACE, PLATE, Verdict
from anonymizer.engine.events import ClipError
from tests.fakes import FixedModule
from tests.test_photo_metadata import real_icc

MAKE = 0x010F


def exif_with_gps() -> "Image.Exif":
    exif = Image.Exif()
    exif[MAKE] = "TestMake"
    exif[ph.GPS_IFD] = {1: "N", 2: (28.0, 6.0, 30.0), 3: "W", 4: (15.0, 25.0, 10.0)}
    return exif


def plate_at(box):
    return [FixedModule(PLATE, [box])]


NOTHING = [FixedModule(PLATE, [])]


class PhotoCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)


class GpsWithoutRecompressing(unittest.TestCase):
    def setUp(self):
        im = Image.fromarray(np.random.default_rng(3).integers(0, 255, (120, 160, 3), dtype=np.uint8))
        exif = exif_with_gps()
        exif[ph.ORIENTATION] = 1
        xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description exif:GPSLatitude="28,6.5N" '
            b'exif:GPSLongitude="15,25.1W" GPano:ProjectionType="equirectangular"/></rdf:RDF></x:xmpmeta>'
        )
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90, exif=exif.tobytes(), xmp=xmp)
        self.data = buf.getvalue()
        self.cleaned, self.changed = ph.jpeg_without_gps(self.data)

    def test_E4_location_removed_and_reported(self):
        before, after = Image.open(io.BytesIO(self.data)), Image.open(io.BytesIO(self.cleaned))
        self.assertTrue(self.changed)
        self.assertTrue(before.getexif().get_ifd(ph.GPS_IFD))
        self.assertFalse(after.getexif().get_ifd(ph.GPS_IFD))

    def test_E4_rest_of_exif_kept_and_pixels_identical(self):
        before, after = Image.open(io.BytesIO(self.data)), Image.open(io.BytesIO(self.cleaned))
        self.assertEqual(after.getexif().get(MAKE), "TestMake")
        np.testing.assert_array_equal(np.array(before), np.array(after))

    def test_E4_xmp_loses_the_location_and_keeps_the_rest(self):
        self.assertNotIn(b"GPSLatitude", self.cleaned)
        self.assertIn(b"GPano:ProjectionType", self.cleaned)

    def test_E4_not_a_jpeg_or_broken_comes_back_unchanged(self):
        self.assertEqual(ph.jpeg_without_gps(b"not a jpeg"), (b"not a jpeg", False))
        self.assertFalse(ph.jpeg_without_gps(self.data[:200])[1])

    def test_E4_gps_under_any_xmp_prefix_or_form_is_found_and_removed(self):
        xmp = (
            b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF><rdf:Description "
            b"xmlns:ns1='http://ns.adobe.com/exif/1.0/' ns1:GPSLatitude='28,6N' "
            b"xmlns:drone-dji='http://www.dji.com/drone-dji/1.0/'><drone-dji:GpsLongitude>-15.4</drone-dji:GpsLongitude>"
            b"<drone-dji:GpsAltitude/></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (120, 90, 60)).save(buf, "JPEG", quality=90)
        plain = buf.getvalue()
        data = plain[:2] + b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp + plain[2:]
        cleaned, changed = ph.jpeg_without_gps(data)
        self.assertTrue(ph.has_gps(data) and changed)
        self.assertFalse(ph.has_gps(cleaned))
        Image.open(io.BytesIO(cleaned)).load()

    def test_E4_xmp_inside_the_exif_is_removed_too(self):
        exif = Image.Exif()
        exif[ph.XMP_IN_EXIF] = b"<x:xmpmeta><rdf:Description exif:GPSLatitude='28,6N'/></x:xmpmeta>"
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, "JPEG", exif=exif.tobytes())
        cleaned, changed = ph.jpeg_without_gps(buf.getvalue())
        self.assertTrue(ph.has_gps(buf.getvalue()) and changed)
        self.assertFalse(ph.has_gps(cleaned))


class ProcessPhotos(PhotoCase):
    def rotated_phone_photo(self) -> str:
        """300x200 upright, stored on its side (orientation 6), text where upright it is (40, 150)-(160, 190); GPS."""
        upright = np.full((300, 200, 3), 200, np.uint8)
        cv2.putText(upright, "AB12", (45, 182), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        stored = np.rot90(upright, k=1)
        exif = Image.Exif()
        exif[ph.ORIENTATION] = 6
        exif[ph.GPS_IFD] = {1: "N", 2: (28.0, 1.0, 2.0)}
        path = self.path("rotated.jpg")
        Image.fromarray(stored[:, :, ::-1]).save(path, "JPEG", quality=95, exif=exif.tobytes())
        return path

    def test_E4_rotated_photo_comes_out_upright_without_gps_and_blurred_in_place(self):
        out = self.path("out.jpg")
        result = ph.process(self.rotated_phone_photo(), out, plate_at((40, 150, 160, 190)))
        saved = Image.open(out)
        self.assertEqual(saved.size, (200, 300))
        self.assertEqual(saved.getexif().get(ph.ORIENTATION), 1)
        self.assertFalse(saved.getexif().get_ifd(ph.GPS_IFD))
        self.assertLess(residual(np.array(saved)[:, :, ::-1], (40, 150, 160, 190)), 1.5)
        self.assertTrue(result.gps_removed)

    def test_E4_jpeg_quality_95_and_source_subsampling(self):
        src = self.path("s.jpg")
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(
            src, "JPEG", quality=80, subsampling=1
        )
        out = self.path("out.jpg")
        ph.process(src, out, plate_at((10, 10, 50, 30)))
        from PIL import JpegImagePlugin

        saved = Image.open(out)
        self.assertEqual(JpegImagePlugin.get_sampling(saved), 1)
        q = saved.quantization[0]
        self.assertLess(sum(q) / len(q), 10)

    def test_E4_png_keeps_its_transparency(self):
        rgba = np.zeros((100, 120, 4), np.uint8)
        rgba[..., :3] = 180
        rgba[..., 3] = np.linspace(0, 255, 120, dtype=np.uint8)[None, :]
        cv2.putText(rgba, "XY", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 3)
        src, out = self.path("alpha.png"), self.path("out.png")
        Image.fromarray(rgba, "RGBA").save(src)
        ph.process(src, out, plate_at((15, 30, 90, 80)))
        saved = np.array(Image.open(out))
        self.assertEqual(saved.shape, (100, 120, 4))
        np.testing.assert_array_equal(saved[..., 3], rgba[..., 3])

    def test_E4_nothing_to_blur_keeps_the_pixels_and_drops_the_gps(self):
        src, out = self.rotated_phone_photo(), self.path("out.jpg")
        result = ph.process(src, out, NOTHING)
        self.assertTrue(result.copied and result.gps_removed)
        np.testing.assert_array_equal(np.array(Image.open(src)), np.array(Image.open(out)))

    def test_E4_nothing_to_blur_and_gps_kept_keeps_the_image_bytes_and_the_location_only(self):
        src, out = self.rotated_phone_photo(), self.path("out.jpg")
        ph.process(src, out, NOTHING, strip_gps=False)
        with open(src, "rb") as a, open(out, "rb") as b:
            before, after = a.read(), b.read()
        self.assertEqual(before[before.index(b"\xff\xda") :], after[after.index(b"\xff\xda") :])
        saved = Image.open(out).getexif()
        self.assertIn(ph.GPS_IFD, saved)
        self.assertEqual((saved.get(ph.ORIENTATION), saved.get(MAKE)), (6, None))

    def test_E4_every_pillow_mode_becomes_the_same_rgb(self):
        for mode in ("RGB", "RGBA", "L", "P", "LA"):
            im = Image.fromarray(np.random.default_rng(1).integers(0, 255, (30, 40, 3), dtype=np.uint8)).convert(mode)
            arr, alpha = ph._to_rgb(im)
            np.testing.assert_array_equal(arr, np.array(im.convert("RGB")))
            self.assertEqual(alpha is not None, mode in ("RGBA", "LA"))

    def test_E4_avif_in_avif_out(self):
        src, out = self.path("photo.avif"), self.path("out.avif")
        upright = np.full((300, 200, 3), 200, np.uint8)
        cv2.putText(upright, "AB12", (45, 182), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        Image.fromarray(upright[:, :, ::-1]).save(src, "AVIF", quality=90)
        ph.process(src, out, plate_at((40, 150, 160, 190)))
        saved = Image.open(out)
        self.assertEqual(saved.format, "AVIF")
        self.assertLess(residual(np.array(saved.convert("RGB"))[:, :, ::-1], (40, 150, 160, 190)), 1.5)

    def test_E4_two_image_jpeg_with_nothing_to_blur_has_no_gps_in_any_image(self):
        rng = np.random.default_rng(7)
        im = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
        second = Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
        buf, extra = io.BytesIO(), io.BytesIO()
        im.save(buf, "MPO", save_all=True, append_images=[second], exif=exif_with_gps().tobytes(), quality=90)
        second.save(extra, "JPEG", exif=exif_with_gps().tobytes())
        src, out = self.path("iphone.jpg"), self.path("out.jpg")
        with open(src, "wb") as f:
            f.write(buf.getvalue() + extra.getvalue())
        self.assertEqual(Image.open(src).format, "MPO")
        result = ph.process(src, out, NOTHING)
        with open(out, "rb") as f:
            self.assertFalse(ph.has_gps(f.read()))
        self.assertTrue(result.gps_removed)

    def test_E4_two_image_jpeg_with_a_blur_keeps_profile_and_quality_and_not_the_camera_data(self):
        rng = np.random.default_rng(7)
        im = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
        second = Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
        icc = real_icc()
        src, out = self.path("iphone.jpg"), self.path("out.jpg")
        im.save(src, "MPO", save_all=True, append_images=[second], exif=exif_with_gps().tobytes(), icc_profile=icc)
        result = ph.process(src, out, plate_at((40, 40, 110, 70)))
        saved = Image.open(out)
        self.assertTrue(any(z.blur for z in result.zones))
        self.assertEqual(
            (saved.format, saved.info.get("icc_profile"), saved.getexif().get(MAKE)), ("JPEG", icc, None)
        )  # the camera make is not on the short list of what an output keeps
        self.assertNotIn(ph.GPS_IFD, saved.getexif())

    def test_E4_png_with_gps_and_nothing_to_blur_loses_the_gps_not_the_pixels(self):
        im = Image.fromarray(np.random.default_rng(7).integers(0, 255, (120, 160, 3), dtype=np.uint8))
        src, out = self.path("plan.png"), self.path("out.png")
        im.save(src, exif=exif_with_gps().tobytes())
        result = ph.process(src, out, NOTHING)
        saved = Image.open(out)
        self.assertEqual(saved.format, "PNG")
        self.assertNotIn(ph.GPS_IFD, saved.getexif())
        np.testing.assert_array_equal(np.array(saved), np.array(im))
        self.assertTrue(result.gps_removed)

    def test_E4_motion_photo_video_tail_is_not_delivered(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (120, 90, 60)).save(buf, "JPEG", quality=90)
        photo = buf.getvalue()
        tail = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64 + b"\xa9xyz+28.1287-015.4200/"
        src, out = self.path("PXL.MP.jpg"), self.path("out.jpg")
        with open(src, "wb") as f:
            f.write(photo + tail)
        result = ph.process(src, out, zones=[])
        with open(out, "rb") as f:
            self.assertEqual(f.read(), photo)
        self.assertIn("photo.warning.trailing_data_removed", result.warnings)

    def test_E4_animation_is_refused_and_nothing_is_written(self):
        src, out = self.path("anim.webp"), self.path("out.webp")
        frames = [Image.new("RGB", (32, 32), (200, 20, 30))]
        Image.new("RGB", (32, 32), (10, 20, 30)).save(
            src, "WEBP", save_all=True, lossless=True, duration=100, append_images=frames
        )
        with self.assertRaises(ClipError) as caught:
            ph.process(src, out, zones=[])
        self.assertEqual(caught.exception.code, "error.photo.several_images")
        self.assertFalse(os.path.exists(out))

    def test_E4_sixteen_bit_grey_png_stays_sixteen_bit_and_unchanged_outside(self):
        g = (np.linspace(0, 65535, 96 * 128).reshape(96, 128)).astype(np.uint16)
        src, out = self.path("g16.png"), self.path("out.png")
        Image.fromarray(g).save(src)
        ph.process(src, out, plate_at((20, 20, 60, 36)))
        with Image.open(out) as saved:
            mode, data = saved.mode, np.asarray(saved).astype(np.int64)
        outside = np.ones(g.shape, bool)
        outside[:60, :90] = False
        self.assertTrue(mode.startswith("I;16"))
        self.assertLess(float(np.abs(data - g.astype(np.int64))[outside].mean()), 1.0)

    def test_E4_cmyk_jpeg_stays_cmyk_with_its_profile(self):
        profile = real_icc()[:16] + b"CMYK" + real_icc()[20:]  # a valid profile that says CMYK
        src, out = self.path("cmyk.jpg"), self.path("out.jpg")
        Image.new("CMYK", (96, 64), (10, 60, 90, 5)).save(src, "JPEG", quality=95, icc_profile=profile)
        ph.process(src, out, plate_at((20, 20, 60, 36)))
        with Image.open(out) as saved:
            self.assertEqual((saved.mode, (saved.info.get("icc_profile") or b"")[16:20]), ("CMYK", b"CMYK"))

    def test_E4_zones_left_visible_in_review_are_not_blurred(self):
        src, out = self.rotated_phone_photo(), self.path("out.jpg")
        first = ph.process(src, self.path("first.jpg"), plate_at((40, 150, 160, 190)))
        result = ph.process(src, out, zones=first.zones, visible={0}, specs=ph.specs_of(plate_at((0, 0, 1, 1))))
        self.assertTrue(result.copied)


class DecidePhoto(unittest.TestCase):
    PLATE_BOX, FACE_BOX = (20, 20, 100, 44), (140, 60, 200, 120)

    def test_E4_faces_first_then_plates_with_the_class_verdicts(self):
        bgr = np.zeros((160, 240, 3), np.uint8)
        modules = [FixedModule(FACE, [self.FACE_BOX], 0.85, 0.75), FixedModule(PLATE, [self.PLATE_BOX], 0.9, 0.9)]
        zones = ph.decide_photo(bgr, modules, unsure="keep")  # v1's rule: the doubtful face is listed and left visible
        self.assertEqual([(z.kind, z.blur) for z in zones], [(FACE, False), (PLATE, True)])
        self.assertEqual(zones[0].judgement.verdict, Verdict.CHECK)
        zones = ph.decide_photo(bgr, modules)  # privacy first: it is blurred, and still listed as doubtful
        self.assertEqual([(z.kind, z.blur) for z in zones], [(FACE, True), (PLATE, True)])
        self.assertEqual(zones[0].judgement.verdict, Verdict.CHECK)


if __name__ == "__main__":
    unittest.main()
