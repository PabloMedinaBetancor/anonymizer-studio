"""H1 of the first review: what an output photo keeps of its metadata is a short list. A MakerNote can hold a full-size preview of the
picture as it was before the blur; serial numbers, owners, comments, XMP, IPTC and Photoshop blocks identify the photographer."""

from __future__ import annotations

import functools
import io
import os
import tempfile
import unittest

import numpy as np

try:
    from PIL import Image, PngImagePlugin

    from anonymizer.engine import photos as ph
    from anonymizer.engine.events import ClipError
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from anonymizer.contract import PLATE
from tests.fakes import FixedModule

BLUR = [FixedModule(PLATE, [(40, 40, 110, 70)])]
NOTHING = [FixedModule(PLATE, [])]
SERIAL, OWNER, COMMENT, ARTIST = (
    "SN-0042-SECRET",
    "Owner Secret Name",
    b"ASCII\x00\x00\x00a secret comment",
    "Artist Secret",
)
XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:creator>XMP-CREATOR-SECRET</dc:creator>'
    b"</rdf:Description></rdf:RDF></x:xmpmeta>"
)
SECRETS = (SERIAL.encode(), OWNER.encode(), b"a secret comment", ARTIST.encode(), b"XMP-CREATOR-SECRET", b"IPTC-SECRET")


def pixels() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 255, (120, 160, 3), dtype=np.uint8)


def preview_jpeg() -> bytes:
    """A small JPEG of the picture before any blur: what a maker note can carry."""
    buf = io.BytesIO()
    Image.fromarray(pixels()[::4, ::4]).save(buf, "JPEG", quality=70)
    return buf.getvalue()


def loaded_exif() -> "Image.Exif":
    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0131], exif[0x013B] = "Maker", "Model", "Software 1.0", ARTIST
    exif[0x010E], exif[0x8298], exif[0x011A], exif[0x0128] = "A description", "(c) someone", 72.0, 2
    inner = exif.get_ifd(ph.EXIF_IFD)
    inner[0x927C] = b"OLYMP\x00\x01\x00" + preview_jpeg()  # the MakerNote with its preview
    inner[0xA431], inner[0xA430], inner[0x9286] = SERIAL, OWNER, COMMENT
    inner[0x9003] = "2026:10:03 10:00:00"
    inner[0x9000], inner[0xA001] = b"0231", 1
    exif[ph.EXIF_IFD] = inner
    exif[ph.XMP_IN_EXIF] = XMP
    exif[ph.GPS_IFD] = {1: "N", 2: (28.0, 6.0, 30.0), 3: "W", 4: (15.0, 25.0, 10.0)}
    return exif


@functools.lru_cache(maxsize=1)
def real_icc() -> bytes:
    """A colour profile that is what it says it is (the one that Pillow builds), as the camera and editing programs write them. LittleCMS
    puts the clock in the header (bytes 24 to 35: the creation date and time), so two calls a second apart gave two different profiles and a
    test that built a file with one and compared with another failed when the second ticked in between (PR-21). The date is fixed and the
    profile is built once."""
    from PIL import ImageCms

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return profile[:24] + bytes(12) + profile[36:]


def segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def add_segments(data: bytes, extra: bytes) -> bytes:
    """The segments after the first (the APP0 or APP1 that Pillow wrote): comments, XMP, IPTC and a JFIF thumbnail."""
    return data[:2] + extra + data[2:]


def full_of_metadata_jpeg(path: str) -> None:
    buf = io.BytesIO()
    Image.fromarray(pixels()).save(buf, "JPEG", quality=92, exif=loaded_exif().tobytes(), icc_profile=real_icc())
    extra = (
        segment(0xFE, b"a secret comment in a COM segment")
        + segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + XMP)
        + segment(0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x0aIPTC-SECRET")
        + segment(0xE0, b"JFXX\x00\x10" + preview_jpeg())
    )
    with open(path, "wb") as f:
        f.write(add_segments(buf.getvalue(), extra))


class Metadata(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def assert_clean(self, out: str, keep_gps: bool = False):
        with open(out, "rb") as f:
            data = f.read()
        self.assertEqual(ph.metadata_leaks(out, keep_gps), [])
        for secret in SECRETS:
            self.assertNotIn(secret, data, secret)
        self.assertNotIn(preview_jpeg()[:200], data)
        if data[:2] == b"\xff\xd8":
            self.assertEqual(data.count(b"\xff\xd8\xff"), 1, "a second JPEG is embedded in the file")
        with Image.open(out) as saved:
            exif = saved.getexif()
            self.assertFalse({0x010F, 0x0110, 0x0131, 0x013B, 0x010E, 0x8298} & set(exif), dict(exif))
            self.assertFalse({0x927C, 0xA431, 0xA430, 0x9286, 0x9003} & set(exif.get_ifd(ph.EXIF_IFD)))
            self.assertNotIn(ph.XMP_IN_EXIF, exif)
            self.assertEqual(ph.GPS_IFD in exif, keep_gps)

    def test_H1_a_blurred_jpeg_keeps_only_the_short_list_and_no_preview_of_the_original(self):
        src, out = self.path("loaded.jpg"), self.path("out.jpg")
        full_of_metadata_jpeg(src)
        with open(src, "rb") as f:
            self.assertGreater(f.read().count(b"\xff\xd8\xff"), 2)  # the fixture does hold previews
        result = ph.process(src, out, BLUR)
        self.assertFalse(result.copied)
        self.assert_clean(out)
        with Image.open(out) as saved:
            exif = saved.getexif()
            self.assertEqual((exif.get(ph.ORIENTATION), exif.get(0x011A), exif.get(0x0128)), (1, 72.0, 2))
            inner = exif.get_ifd(ph.EXIF_IFD)
            self.assertEqual((inner.get(0xA001), inner.get(0xA002), inner.get(0xA003)), (1, 160, 120))
            self.assertEqual(saved.info.get("icc_profile"), real_icc())

    def test_H1_a_jpeg_with_nothing_to_blur_is_rewritten_by_segments_and_its_image_data_is_untouched(self):
        src, out = self.path("loaded.jpg"), self.path("out.jpg")
        full_of_metadata_jpeg(src)
        result = ph.process(src, out, NOTHING)
        self.assertTrue(result.copied and result.gps_removed)
        self.assert_clean(out)
        with open(src, "rb") as a, open(out, "rb") as b:
            before, after = a.read(), b.read()
        self.assertEqual(before[before.rindex(b"\xff\xda") :], after[after.rindex(b"\xff\xda") :])
        np.testing.assert_array_equal(np.array(Image.open(src)), np.array(Image.open(out)))
        with Image.open(out) as saved:
            self.assertEqual(saved.info.get("icc_profile"), real_icc())

    def test_H1_keeping_the_location_keeps_the_gps_block_and_nothing_else_of_the_metadata(self):
        src = self.path("loaded.jpg")
        full_of_metadata_jpeg(src)
        for name, modules in (("blurred.jpg", BLUR), ("copied.jpg", NOTHING)):
            out = self.path(name)
            ph.process(src, out, modules, strip_gps=False)
            self.assert_clean(out, keep_gps=True)
            self.assertEqual(Image.open(out).getexif().get_ifd(ph.GPS_IFD).get(1), "N")

    def test_H1_png_with_exif_text_and_xmp(self):
        src = self.path("loaded.png")
        info = PngImagePlugin.PngInfo()
        info.add_text("Author", ARTIST)
        info.add_itxt("XML:com.adobe.xmp", XMP.decode())
        info.add_text("Comment", "a secret comment")
        Image.fromarray(pixels()).save(src, "PNG", exif=loaded_exif().tobytes(), pnginfo=info)
        blurred = self.path("blurred.png")
        ph.process(src, blurred, BLUR)
        self.assert_clean(blurred)
        copied = self.path("copied.png")
        result = ph.process(src, copied, NOTHING)
        self.assertFalse(result.copied)  # a PNG is always written again: nothing the analysers do not know can stay
        self.assert_clean(copied)
        np.testing.assert_array_equal(np.array(Image.open(copied)), pixels())

    def test_H1_a_png_with_nothing_but_pixels_is_written_again_with_the_same_pixels(self):
        src, out = self.path("plain.png"), self.path("out.png")
        Image.fromarray(pixels()).save(src, "PNG")
        result = ph.process(src, out, NOTHING)
        self.assertFalse(result.copied)
        self.assertEqual(result.warnings, [])  # no loss: nothing to warn about
        np.testing.assert_array_equal(np.array(Image.open(out)), pixels())

    def test_H1_webp_with_exif_and_xmp(self):
        src = self.path("loaded.webp")
        Image.fromarray(pixels()).save(src, "WEBP", lossless=True, exif=loaded_exif().tobytes(), xmp=XMP)
        for name, modules in (("blurred.webp", BLUR), ("copied.webp", NOTHING)):
            out = self.path(name)
            ph.process(src, out, modules)
            self.assert_clean(out)
        np.testing.assert_array_equal(np.array(Image.open(self.path("copied.webp"))), pixels())

    def test_H1_tiff_with_photoshop_iptc_and_xmp_blocks(self):
        from PIL import TiffImagePlugin

        src = self.path("loaded.tif")
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[34377] = b"8BIM\x04\x0c" + preview_jpeg()
        tags[33723] = b"IPTC-SECRET"
        tags[700] = XMP
        Image.fromarray(pixels()).save(src, "TIFF", tiffinfo=tags)
        with Image.open(src) as check:
            self.assertTrue({34377, 33723, 700} <= set(check.tag_v2))
        for name, modules in (("blurred.tif", BLUR), ("copied.tif", NOTHING)):
            out = self.path(name)
            ph.process(src, out, modules)
            self.assert_clean(out)
            with Image.open(out) as saved:
                self.assertFalse({34377, 33723, 700} & set(saved.tag_v2))
        np.testing.assert_array_equal(np.array(Image.open(self.path("copied.tif"))), pixels())

    def test_H1_a_tiff_with_only_its_own_structure_and_one_with_an_iptc_block_are_both_written_again(self):
        from PIL import TiffImagePlugin

        plain, out = self.path("plain.tif"), self.path("out.tif")
        Image.fromarray(pixels()).save(plain, "TIFF", compression="tiff_lzw")
        self.assertFalse(ph.process(plain, out, NOTHING).copied)
        np.testing.assert_array_equal(np.array(Image.open(out)), pixels())
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33723] = b"IPTC-SECRET"
        tagged, again = self.path("tagged.tif"), self.path("again.tif")
        Image.fromarray(pixels()).save(tagged, "TIFF", tiffinfo=tags)
        result = ph.process(tagged, again, NOTHING)
        self.assertFalse(result.copied)
        self.assertEqual(ph.metadata_leaks(again), [])

    def test_H1_avif_with_exif(self):
        src = self.path("loaded.avif")
        try:
            Image.fromarray(pixels()).save(src, "AVIF", quality=90, exif=loaded_exif().tobytes())
        except (KeyError, ValueError, OSError) as e:
            self.skipTest(f"no AVIF writer here: {e}")
        out = self.path("blurred.avif")
        ph.process(src, out, BLUR)
        self.assert_clean(out)

    def test_H1_the_check_refuses_a_photo_that_still_carries_a_preview(self):
        leaky = self.path("leaky.jpg")
        full_of_metadata_jpeg(leaky)
        leaks = ph.metadata_leaks(leaky)
        self.assertTrue(any("segment" in leak for leak in leaks), leaks)
        self.assertTrue(any("0x927c" in leak for leak in leaks), leaks)
        with self.assertRaises(ClipError) as caught:
            ph._check_clean(leaky, False)
        self.assertEqual(caught.exception.code, "error.photo.metadata_not_removed")
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "JPEG", exif=loaded_exif().tobytes())
        only_exif = self.path("only_exif.jpg")
        with open(only_exif, "wb") as f:
            f.write(buf.getvalue())
        self.assertTrue(any("0xa431" in leak for leak in ph.metadata_leaks(only_exif)))

    def test_H1_a_comment_of_an_opened_picture_is_not_written_back_by_save(self):
        src = self.path("loaded.jpg")
        full_of_metadata_jpeg(src)
        with Image.open(src) as im:
            im.load()
            self.assertEqual(im.info.get("comment"), b"a secret comment in a COM segment")
            out = self.path("saved.jpg")
            ph.save(im, out, "JPEG", im.getexif(), 2, None, dict(im.info), True)
        with open(out, "rb") as f:
            self.assertNotIn(b"a secret comment", f.read())
        self.assertEqual(ph.metadata_leaks(out), [])

    def test_H1_a_rotated_copy_keeps_its_orientation_and_a_rotated_resave_does_not(self):
        exif = Image.Exif()
        exif[ph.ORIENTATION] = 6
        src = self.path("turned.jpg")
        Image.fromarray(pixels()).save(src, "JPEG", exif=exif.tobytes())
        copied = self.path("copied.jpg")
        ph.process(src, copied, NOTHING)
        self.assertEqual(Image.open(copied).getexif().get(ph.ORIENTATION), 6)
        blurred = self.path("blurred.jpg")
        ph.process(src, blurred, [FixedModule(PLATE, [(10, 10, 60, 40)])])
        saved = Image.open(blurred)
        self.assertEqual((saved.getexif().get(ph.ORIENTATION), saved.size), (1, (120, 160)))


class PublicPhotos(unittest.TestCase):
    """The JPEGs of the public benchmark that carry an image inside a metadata segment (the EXIF thumbnail of the cameras), from
    bench/data (only in the main checkout: skipped where it is not)."""

    def test_H1_public_photos_with_an_embedded_image_come_out_clean_blurred_or_not(self):
        import glob

        root = os.environ.get("ANONYMIZER_BENCH_DATA", "")
        paths = sorted(glob.glob(os.path.join(root, "plates", "openalpr-eu", "*.jpg")))
        carrying = [p for p in paths if self.embedded_image(p)]
        if not carrying:
            self.skipTest("the public benchmark photos are not here")
        self.assertGreaterEqual(len(carrying), 20)
        with tempfile.TemporaryDirectory() as tmp:
            for n, path in enumerate(carrying[:40]):
                for kind, modules in (("blurred", [FixedModule(PLATE, [(5, 5, 40, 25)])]), ("copied", NOTHING)):
                    out = os.path.join(tmp, f"{kind}{n}.jpg")
                    ph.process(path, out, modules)
                    with open(out, "rb") as f:
                        data = f.read()
                    self.assertEqual(data.count(b"\xff\xd8\xff"), 1, (kind, os.path.basename(path)))
                    self.assertEqual(ph.metadata_leaks(out), [], (kind, os.path.basename(path)))

    @staticmethod
    def embedded_image(path: str) -> bool:
        with open(path, "rb") as f:
            data = f.read()
        i = 2
        while i + 4 < len(data) and data[i] == 0xFF and data[i + 1] not in (0xDA, 0xD9):
            length = int.from_bytes(data[i + 2 : i + 4], "big")
            if data[i + 1] == 0xE1 and b"\xff\xd8\xff" in data[i + 4 : i + 2 + length]:
                return True
            i += 2 + length
        return False


if __name__ == "__main__":
    unittest.main()
