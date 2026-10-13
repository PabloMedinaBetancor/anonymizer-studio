"""Review 3, R3-1 to R3-4: a photo in a format without loss (PNG, TIFF, lossless WebP) with nothing to blur is written again from the
decoded picture, always. These tests show what that costs: nothing. The pixels are the same in every mode, with the transparency, the
orientation (the picture comes out upright, as when something is blurred), the resolution and a valid colour profile."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

import numpy as np

try:
    from PIL import Image, ImageOps

    from anonymizer.engine import photos as ph
    from anonymizer.engine.containers import container_leaks
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from anonymizer.contract import PLATE
from tests.fakes import FixedModule
from tests.test_photo_metadata import real_icc

NOTHING = [FixedModule(PLATE, [])]


def noise(height=64, width=96, channels=None, dtype=np.uint8, high=255, seed=3):
    shape = (height, width) if channels is None else (height, width, channels)
    return np.random.default_rng(seed).integers(0, high, shape).astype(dtype)


def images():
    """(name, picture, formats it can be saved in) for every mode that the formats hold."""
    palette = Image.fromarray(noise(), "L").convert("P")
    transparent = palette.copy()
    transparent.info["transparency"] = bytes([0, 128, 255]) + bytes([255] * 253)
    cutout = Image.fromarray(noise(channels=3), "RGB")
    cutout.info["transparency"] = (10, 20, 30)
    alpha = noise(channels=4)
    alpha[:8, :8, 3] = 0  # a transparent corner with colour under it
    return [
        ("bilevel", Image.fromarray(noise() > 127), ("PNG", "TIFF")),
        ("grey", Image.fromarray(noise(), "L"), ("PNG", "TIFF")),
        ("grey+alpha", Image.fromarray(noise(channels=2), "LA"), ("PNG",)),
        ("palette", palette, ("PNG", "TIFF")),
        ("palette+transparency", transparent, ("PNG",)),
        ("rgb", Image.fromarray(noise(channels=3), "RGB"), ("PNG", "TIFF", "WEBP")),
        ("rgb+colour key", cutout, ("PNG",)),
        ("rgba", Image.fromarray(alpha, "RGBA"), ("PNG", "TIFF", "WEBP")),
        ("cmyk", Image.fromarray(noise(channels=4), "CMYK"), ("TIFF",)),
        ("16 bit grey", Image.fromarray(noise(dtype=np.uint16, high=65535)), ("PNG", "TIFF")),
        ("float", Image.fromarray(noise().astype(np.float32) / 7.0), ("TIFF",)),
    ]


def what_is_seen(im) -> tuple:
    """The mode and the bytes of a picture as someone looking at it sees them (transparency made an alpha channel)."""
    if "transparency" in im.info:
        im = im.convert("RGBA")
    return im.mode, im.tobytes()


class SamePixels(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def test_R3_every_mode_of_every_lossless_format_comes_out_with_the_same_pixels(self):
        ran = 0
        for name, picture, formats in images():
            for fmt in formats:
                src = os.path.join(self.dir, f"{name.replace(' ', '_').replace('+', '_')}.{fmt.lower()}")
                kw = {"lossless": True} if fmt == "WEBP" else {}
                try:
                    picture.save(src, fmt, **kw)
                except (KeyError, ValueError, OSError):
                    continue  # this Pillow cannot write the mode in this format: no such file exists to be tested
                out = src + ".out"
                result = ph.process(src, out, NOTHING)
                self.assertFalse(result.copied, (name, fmt))
                self.assertEqual(ph.metadata_leaks(out), [], (name, fmt))
                with Image.open(src) as a, Image.open(out) as b:
                    a.load()
                    b.load()
                    seen_before, seen_after = what_is_seen(a), what_is_seen(b)
                self.assertEqual(seen_before[0], seen_after[0], (name, fmt, "mode"))
                if fmt == "WEBP" and name == "rgba":
                    # a pixel that is fully transparent shows nothing whatever colour it holds: compare what is visible
                    self.assertEqual(self.visible(seen_before), self.visible(seen_after), (name, fmt))
                else:
                    self.assertEqual(seen_before[1], seen_after[1], (name, fmt, "pixels"))
                ran += 1
        self.assertGreaterEqual(ran, 14)

    @staticmethod
    def visible(seen: tuple) -> bytes:
        data = np.frombuffer(seen[1], np.uint8).reshape(-1, 4).copy()
        data[data[:, 3] == 0] = 0
        return data.tobytes()

    def test_R3_the_orientation_the_resolution_and_a_valid_profile_survive(self):
        profile = real_icc()
        for fmt in ("PNG", "TIFF"):
            src = os.path.join(self.dir, f"o.{fmt.lower()}")
            exif = Image.Exif()
            exif[0x112] = 6  # rotate 270 to be upright
            picture = Image.fromarray(noise(channels=3), "RGB")
            extra = {"exif": exif.tobytes()} if fmt == "PNG" else {}
            picture.save(src, fmt, dpi=(300, 300), icc_profile=profile, **extra)
            out = src + ".out"
            ph.process(src, out, NOTHING)
            with Image.open(src) as a, Image.open(out) as b:
                upright = ImageOps.exif_transpose(a) if fmt == "PNG" else a
                self.assertEqual(upright.size, b.size, fmt)
                if fmt == "PNG":
                    self.assertEqual(upright.tobytes(), b.tobytes())
                self.assertEqual(b.info.get("icc_profile"), profile, fmt)
                self.assertEqual(tuple(round(v) for v in b.info.get("dpi", (0, 0))), (300, 300), fmt)

    def test_R3_a_png_with_a_second_directory_inside_its_exif_chunk_is_clean_after_the_copy_path(self):
        # what exiftool or a converter leaves: the camera's thumbnail in IFD1 of the eXIf chunk (probe A1 of the review)
        from tests.test_photo_containers import exif_with_thumbnail, png_chunk, png_with_chunk

        secret = b"SERIAL-1234"
        data = png_with_chunk(png_chunk(b"eXIf", exif_with_thumbnail(b"\xff\xd8\xff" + secret)))
        src = os.path.join(self.dir, "x.png")
        with open(src, "wb") as f:
            f.write(data)
        out = src + ".out"
        ph.process(src, out, NOTHING)
        with open(out, "rb") as f:
            self.assertNotIn(secret, f.read())

    def test_R3_a_webp_with_loss_and_nothing_extra_is_still_a_byte_copy(self):
        buf = io.BytesIO()
        Image.fromarray(noise(channels=3), "RGB").save(buf, "WEBP", quality=80)
        src = os.path.join(self.dir, "l.webp")
        with open(src, "wb") as f:
            f.write(buf.getvalue())
        out = src + ".out"
        self.assertTrue(ph.process(src, out, NOTHING).copied)
        with open(out, "rb") as f:
            self.assertEqual(f.read(), buf.getvalue())
        self.assertEqual(container_leaks(buf.getvalue()), [])


if __name__ == "__main__":
    unittest.main()
