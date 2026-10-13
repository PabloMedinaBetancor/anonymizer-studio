"""Review 2, N2 and N3: what Pillow does not show. A COM segment between the scans of a progressive JPEG, a thumbnail inside the JFIF
header, a private PNG chunk, an extra RIFF chunk of a WebP, TIFF tags of the owner and a thumbnail item of an AVIF all came out in the
copy of a photo with nothing to blur, and the check that runs on every written photo did not see them either. The check reads the
container from its bytes now; a chunk put by hand into a clean output must make it fail."""

from __future__ import annotations

import io
import os
import struct
import tempfile
import unittest
import zlib
from typing import Tuple

import numpy as np

try:
    from PIL import Image, TiffImagePlugin

    from anonymizer.engine import containers
    from anonymizer.engine import photos as ph
    from anonymizer.engine.events import ClipError
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from anonymizer.contract import PLATE
from tests.fakes import FixedModule

BLUR = [FixedModule(PLATE, [(40, 40, 110, 70)])]
NOTHING = [FixedModule(PLATE, [])]
SECRET = b"SECRET-SERIAL-9999"


def pixels() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 255, (120, 160, 3), dtype=np.uint8)


def preview() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels()[::4, ::4]).save(buf, "JPEG")
    return buf.getvalue()


def payload() -> bytes:
    return preview() + SECRET


def segment(marker: int, body: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(body) + 2).to_bytes(2, "big") + body


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def png_with_chunk(extra: bytes) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels()).save(buf, "PNG")
    data = buf.getvalue()
    idat = data.find(b"IDAT") - 4
    return data[:idat] + extra + data[idat:]


def webp_with_chunk(kind: bytes, body: bytes) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels()).save(buf, "WEBP", quality=90)
    data = buf.getvalue()
    chunk = kind + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")
    return data[:4] + struct.pack("<I", len(data) - 8 + len(chunk)) + data[8:] + chunk


def progressive_jpeg_with_comment_between_scans() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels()).save(buf, "JPEG", progressive=True, quality=90)
    data = buf.getvalue()
    scans = [i for i, (_, _) in enumerate(zip(data, data[1:], strict=False)) if data[i] == 0xFF and data[i + 1] == 0xDA]
    assert len(scans) > 1, "Pillow wrote a single scan"
    return data[: scans[1]] + segment(0xFE, payload()) + data[scans[1] :]


def jpeg_with_jfif_thumbnail() -> Tuple[bytes, bytes]:
    buf = io.BytesIO()
    Image.fromarray(pixels()).save(buf, "JPEG", quality=90)
    data = buf.getvalue()
    thumb = np.asarray(Image.fromarray(pixels()).resize((8, 8))).tobytes()
    jfif = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x08\x08" + thumb
    first = data.find(b"\xff\xe0")
    length = int.from_bytes(data[first + 2 : first + 4], "big")
    return data[:first] + segment(0xE0, jfif) + data[first + 2 + length :], thumb


def tiff_with(tags: dict, types: dict | None = None) -> bytes:
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    for tag, value in tags.items():
        if types and tag in types:
            ifd.tagtype[tag] = types[tag]
        ifd[tag] = value
    buf = io.BytesIO()
    Image.fromarray(pixels()).save(buf, "TIFF", tiffinfo=ifd)
    return buf.getvalue()


class FileCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def anonymise(self, name: str, data: bytes, zones=NOTHING) -> bytes:
        source = self.write(name, data)
        target = source + ".out"
        ph.process(source, target, zones)  # raises when the written file still has a leak
        with open(target, "rb") as f:
            out = f.read()
        self.assertEqual(ph.metadata_leaks(target), [])
        self.assertNotIn(SECRET, out)
        self.assertNotIn(preview()[:200], out)
        return out


class Containers(FileCase):
    def test_N2_a_comment_between_the_scans_of_a_progressive_jpeg_does_not_come_out(self):
        data = progressive_jpeg_with_comment_between_scans()
        self.assertTrue(ph.metadata_leaks(self.write("src.jpg", data)), "the check must see it in the source")
        for zones in (NOTHING, BLUR):
            out = self.anonymise("p.jpg", data, zones)
            if zones is NOTHING:
                self.assertGreater(
                    out.count(b"\xff\xda"), 1, "nothing to blur: the image data is a copy, still progressive"
                )

    def test_N2_a_jfif_thumbnail_does_not_come_out_and_the_header_says_there_is_none(self):
        data, thumb = jpeg_with_jfif_thumbnail()
        self.assertTrue(ph.metadata_leaks(self.write("src.jpg", data)))
        out = self.anonymise("j.jpg", data)
        self.assertNotIn(thumb, out)
        header = out[out.find(b"JFIF\x00") :][:14]
        self.assertEqual(header[12:14], b"\x00\x00")

    def test_N2_a_private_png_chunk_does_not_come_out(self):
        data = png_with_chunk(png_chunk(b"prVW", payload()))
        self.assertTrue(ph.metadata_leaks(self.write("src.png", data)))
        for zones in (NOTHING, BLUR):
            self.assertNotIn(b"prVW", self.anonymise("c.png", data, zones))

    def test_N2_data_after_the_end_of_a_png_does_not_come_out(self):
        data = png_with_chunk(b"") + payload()
        self.assertTrue(ph.metadata_leaks(self.write("src.png", data)))
        self.anonymise("t.png", data)

    def test_N2_an_extra_riff_chunk_of_a_webp_does_not_come_out(self):
        data = webp_with_chunk(b"prVW", payload())
        self.assertTrue(ph.metadata_leaks(self.write("src.webp", data)))
        for zones in (NOTHING, BLUR):
            self.assertNotIn(b"prVW", self.anonymise("w.webp", data, zones))

    def test_N2_tiff_tags_of_the_owner_do_not_come_out(self):
        data = tiff_with(
            {
                269: "DOCNAME-" + SECRET.decode(),
                285: "PAGENAME-" + SECRET.decode(),
                40093: SECRET.decode().encode("utf-16-le"),
            },
            {40093: 1},
        )
        self.assertTrue(ph.metadata_leaks(self.write("src.tif", data)))
        out = self.anonymise("t.tif", data)
        self.assertNotIn(b"DOCNAME", out)
        self.assertNotIn(SECRET.decode().encode("utf-16-le"), out)

    def test_N2_the_check_fails_when_a_chunk_is_put_by_hand_into_a_clean_output(self):
        png = self.anonymise("c.png", png_with_chunk(b""))
        at = png.find(b"IDAT") - 4
        bad = self.write("bad.png", png[:at] + png_chunk(b"prVW", payload()) + png[at:])
        with self.assertRaises(ClipError) as caught:
            ph._check_clean(bad, False)
        self.assertEqual(caught.exception.code, "error.photo.metadata_not_removed")
        webp = self.anonymise("c.webp", webp_with_chunk(b"EXIF", b""))
        riff = webp[:4] + struct.pack("<I", len(webp) - 8 + 8 + len(payload()) + (len(payload()) & 1)) + webp[8:]
        riff += b"prVW" + struct.pack("<I", len(payload())) + payload() + (b"\x00" if len(payload()) & 1 else b"")
        with self.assertRaises(ClipError):
            ph._check_clean(self.write("bad.webp", riff), False)

    def test_N2_the_check_fails_for_a_comment_between_scans_and_data_after_the_image_in_a_clean_jpeg(self):
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "JPEG", progressive=True, quality=90)
        clean = buf.getvalue()
        self.assertEqual(ph.metadata_leaks(self.write("ok.jpg", clean)), [])
        self.assertTrue(ph.metadata_leaks(self.write("com.jpg", progressive_jpeg_with_comment_between_scans())))
        self.assertTrue(ph.metadata_leaks(self.write("tail.jpg", clean + payload())))

    def test_N2_a_webp_check_fails_for_a_chunk_put_in_by_hand(self):
        self.assertEqual(ph.metadata_leaks(self.write("ok.webp", self._plain_webp())), [])
        self.assertTrue(ph.metadata_leaks(self.write("x.webp", webp_with_chunk(b"prVW", payload()))))

    @staticmethod
    def _plain_webp() -> bytes:
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "WEBP", quality=90)
        return buf.getvalue()

    def test_N3_a_whitelisted_tag_with_a_value_that_is_not_what_it_should_be_is_dropped_not_refused(self):
        exif = Image.Exif()
        inner = exif.get_ifd(ph.EXIF_IFD)
        inner[0x9101] = b"\x01\x02\x03\x00" + payload()  # ComponentsConfiguration is four bytes
        inner[0x9000] = b"0231"
        exif[ph.EXIF_IFD] = inner
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "JPEG", exif=exif.tobytes())
        for zones in (NOTHING, BLUR):
            out = self.anonymise("blob.jpg", buf.getvalue(), zones)
            with Image.open(io.BytesIO(out)) as saved:
                kept = saved.getexif().get_ifd(ph.EXIF_IFD)
            self.assertNotIn(0x9101, kept)
            self.assertEqual(kept.get(0x9000), b"0231")

    def test_N3_a_value_that_fits_is_kept(self):
        self.assertTrue(ph._value_fits(0x9101, b"\x01\x02\x03\x00"))
        self.assertTrue(ph._value_fits(0xA001, 1))
        self.assertFalse(ph._value_fits(0xA001, 7))
        self.assertFalse(ph._value_fits(0x9000, b"0231" * 500))


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def full_box(kind: bytes, version: int, body: bytes) -> bytes:
    return box(kind, bytes([version, 0, 0, 0]) + body)


def infe(item_id: int, kind: bytes) -> bytes:
    return full_box(b"infe", 2, struct.pack(">HH4s", item_id, 0, kind) + b"\x00")


def avif(items: dict, refs: bytes = b"", extra_box: bytes = b"") -> bytes:
    iinf = full_box(b"iinf", 0, struct.pack(">H", len(items)) + b"".join(infe(i, k) for i, k in items.items()))
    meta = full_box(
        b"meta", 0, full_box(b"hdlr", 0, b"\x00" * 21) + full_box(b"pitm", 0, b"\x00\x01") + iinf + refs + extra_box
    )
    return box(b"ftyp", b"avif\x00\x00\x00\x00avifmif1") + meta + box(b"mdat", b"\x00" * 8)


class AvifContainer(unittest.TestCase):
    def test_N2_an_avif_with_its_picture_and_an_exif_item_is_clean(self):
        refs = full_box(b"iref", 0, box(b"cdsc", struct.pack(">HHH", 2, 1, 1)))
        self.assertEqual(containers.avif_leaks(avif({1: b"av01", 2: b"Exif"}, refs)), [])

    def test_N2_a_thumbnail_item_an_xmp_item_and_an_extra_box_are_leaks(self):
        thumb = full_box(b"iref", 0, box(b"thmb", struct.pack(">HHH", 2, 1, 1)))
        leaks = containers.avif_leaks(avif({1: b"av01", 2: b"av01"}, thumb))
        self.assertIn("AVIF thumbnail", leaks)
        self.assertTrue(any("mime" in leak for leak in containers.avif_leaks(avif({1: b"av01", 2: b"mime"}))))
        self.assertTrue(
            any("udta" in leak for leak in containers.avif_leaks(avif({1: b"av01"}, b"", box(b"udta", b"x"))))
        )
        self.assertTrue(
            any("free" in leak for leak in containers.avif_leaks(avif({1: b"av01"}) + box(b"free", payload())))
        )

    def test_N2_a_file_that_is_not_understood_is_a_leak_not_a_pass(self):
        self.assertTrue(containers.avif_leaks(b"\x00\x00\x00\x20ftypavif" + b"\x01" * 10))


def exif_with_thumbnail(thumb: bytes) -> bytes:
    """An EXIF blob (TIFF header) the way a converter leaves it: IFD0 with the orientation, then IFD1 with the camera's JPEG thumbnail."""
    head = b"II*\x00" + struct.pack("<I", 8)
    ifd0 = (
        struct.pack("<H", 1)
        + struct.pack("<HHI", 0x112, 3, 1)
        + struct.pack("<HH", 1, 0)
        + struct.pack("<I", 8 + 2 + 12 + 4)
    )
    where = 8 + 2 + 12 + 4 + 2 + 2 * 12 + 4
    ifd1 = (
        struct.pack("<H", 2)
        + struct.pack("<HHII", 0x201, 4, 1, where)
        + struct.pack("<HHII", 0x202, 4, 1, len(thumb))
        + struct.pack("<I", 0)
    )
    return head + ifd0 + ifd1 + thumb


def plain_exif() -> bytes:
    exif = Image.Exif()
    exif[0x112] = 1
    raw = exif.tobytes()
    return raw[6:] if raw.startswith(b"Exif\x00\x00") else raw


class ThumbnailInsideTheExifBlob(FileCase):
    """Review 3, R3-1: the EXIF of a PNG, a WebP or an AVIF is one blob that Pillow shows only in part."""

    def test_R3_1_a_png_exif_with_a_thumbnail_directory_does_not_come_out(self):
        data = png_with_chunk(png_chunk(b"eXIf", exif_with_thumbnail(payload())))
        self.assertTrue(
            ph.metadata_leaks(self.write("src.png", data)), "the check must see the thumbnail in the source"
        )
        for zones in (NOTHING, BLUR):
            self.anonymise("e.png", data, zones)

    def test_R3_1_a_webp_exif_with_a_thumbnail_directory_does_not_come_out(self):
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "WEBP", quality=90, exif=exif_with_thumbnail(payload()))
        self.assertTrue(ph.metadata_leaks(self.write("src.webp", buf.getvalue())))
        for zones in (NOTHING, BLUR):
            self.anonymise("e.webp", buf.getvalue(), zones)

    def test_R3_1_the_check_fails_when_the_blob_is_put_by_hand_into_a_clean_output(self):
        for blob in (exif_with_thumbnail(payload()), plain_exif() + payload()):
            png = self.anonymise("c.png", png_with_chunk(b""))
            at = png.find(b"IDAT") - 4
            bad = self.write("bad.png", png[:at] + png_chunk(b"eXIf", blob) + png[at:])
            with self.assertRaises(ClipError):
                ph._check_clean(bad, False)

    def test_R3_1_the_exif_walker_reads_directories_values_and_what_is_left_over(self):
        self.assertEqual(containers.exif_blob_leaks(plain_exif()), [])
        self.assertEqual(containers.exif_blob_leaks(b"Exif\x00\x00" + plain_exif()), [])
        self.assertTrue(any("thumbnail" in leak for leak in containers.exif_blob_leaks(exif_with_thumbnail(b"abc"))))
        self.assertTrue(
            any("no directory points" in leak for leak in containers.exif_blob_leaks(plain_exif() + b"hidden"))
        )
        self.assertEqual(containers.exif_blob_leaks(plain_exif() + b"\x00\x00\x00"), [])  # padding
        self.assertTrue(containers.exif_blob_leaks(b"not a tiff at all"))


class WholeTiffStructure(FileCase):
    def test_R3_2_bytes_appended_after_the_last_strip_do_not_come_out(self):
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "TIFF")
        data = buf.getvalue() + payload()
        self.assertTrue(ph.metadata_leaks(self.write("src.tif", data)))
        self.anonymise("t.tif", data)

    def test_R3_2_a_region_that_no_directory_points_at_is_a_leak_wherever_it_sits(self):
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "TIFF")
        clean = buf.getvalue()
        self.assertEqual(containers.tiff_leaks(clean), [])
        self.assertTrue(ph.metadata_leaks(self.write("gap.tif", clean + b"\x00" * 7 + SECRET)))

    def test_R3_2_the_check_fails_when_bytes_are_put_by_hand_into_a_clean_tiff_output(self):
        out = self.anonymise("c.tif", self.tiff_bytes())
        with self.assertRaises(ClipError):
            ph._check_clean(self.write("bad.tif", out + payload()), False)

    @staticmethod
    def tiff_bytes() -> bytes:
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "TIFF")
        return buf.getvalue()


class JpegWhiteList(FileCase):
    def jpeg(self) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(pixels()).save(buf, "JPEG", quality=90)
        return buf.getvalue()

    def test_R3_3_a_line_count_segment_with_a_payload_is_cut_to_its_two_bytes(self):
        data = self.jpeg()
        at = data.find(b"\xff\xdb")
        stuffed = data[:at] + b"\xff\xdc" + struct.pack(">H", len(payload()) + 2) + payload() + data[at:]
        Image.open(io.BytesIO(stuffed)).load()  # the decoder reads it
        self.assertTrue(ph.metadata_leaks(self.write("src.jpg", stuffed)))
        out = self.anonymise("d.jpg", stuffed)
        self.assertIn(b"\xff\xdc\x00\x04", out)

    def test_R3_3_markers_that_are_not_on_the_list_do_not_come_out(self):
        data = self.jpeg()
        at = data.find(b"\xff\xdb")
        for marker in (0xF0, 0xFD, 0x60, 0xC8, 0xDE, 0xDF):
            stuffed = data[:at] + bytes([0xFF, marker]) + struct.pack(">H", len(SECRET) + 2) + SECRET + data[at:]
            self.assertTrue(ph.metadata_leaks(self.write("src.jpg", stuffed)), hex(marker))
            try:
                Image.open(io.BytesIO(stuffed)).load()
            except Exception:
                continue  # not even decodable: refused earlier
            self.assertNotIn(SECRET, self.anonymise("m.jpg", stuffed))

    def test_R3_3_a_clean_progressive_and_baseline_jpeg_rewrite_to_the_same_bytes(self):
        for progressive in (False, True):
            buf = io.BytesIO()
            Image.fromarray(pixels()).save(buf, "JPEG", quality=90, progressive=progressive, optimize=True)
            out = self.anonymise(f"p{progressive}.jpg", buf.getvalue())
            self.assertEqual(containers.jpeg_leaks(out), [])
            np.testing.assert_array_equal(
                np.array(Image.open(io.BytesIO(out))), np.array(Image.open(io.BytesIO(buf.getvalue())))
            )


class IccProfile(FileCase):
    def test_PR21_the_test_profile_does_not_depend_on_the_clock(self):
        from tests.test_photo_metadata import real_icc

        self.assertEqual(real_icc()[24:36], bytes(12))
        self.assertEqual(containers.icc_leaks(real_icc()), [])

    def test_R3_4_a_profile_that_carries_more_than_colour_is_dropped_the_real_one_is_kept(self):
        from tests.test_photo_metadata import real_icc

        good = real_icc()
        self.assertEqual(containers.icc_leaks(good), [])
        for bad in (good + payload(), good[:-4], good[:36] + b"xxxx" + good[40:]):
            self.assertTrue(containers.icc_leaks(bad))
        im = Image.fromarray(pixels())
        for fmt, name in (("PNG", "i.png"), ("JPEG", "i.jpg"), ("TIFF", "i.tif")):
            path = os.path.join(self.dir, name)
            im.save(path, fmt, icc_profile=good + SECRET + preview())
            with open(path, "rb") as f:
                out = self.anonymise(name, f.read())
            with Image.open(io.BytesIO(out)) as saved:
                self.assertIsNone(saved.info.get("icc_profile"), fmt)
            self.assertNotIn(SECRET, out)
            kept_path = os.path.join(self.dir, "k" + name)
            im.save(kept_path, fmt, icc_profile=good)
            with open(kept_path, "rb") as f:
                kept = self.anonymise("k" + name, f.read())
            with Image.open(io.BytesIO(kept)) as saved:
                self.assertEqual(saved.info.get("icc_profile"), good, fmt)


class AvifBehaviour(unittest.TestCase):
    def test_R3_7_a_real_avif_from_pillow_has_nothing_to_remove_and_its_two_info_keys_are_not_metadata(self):
        with open(os.path.join(os.path.dirname(__file__), "data", "pillow-plain.avif"), "rb") as f:
            data = f.read()
        self.assertEqual(containers.container_leaks(data), [])
        self.assertLessEqual({"timestamp", "duration"}, ph.TECHNICAL_INFO)

    def test_R3_1_the_exif_item_of_an_avif_is_read_and_a_thumbnail_directory_in_it_is_a_leak(self):
        for blob, leaky in ((plain_exif(), False), (exif_with_thumbnail(payload()), True)):
            leaks = containers.avif_leaks(avif_with_exif(blob))
            self.assertEqual(any("EXIF" in leak for leak in leaks), leaky, leaks)


def avif_with_exif(blob: bytes) -> bytes:
    """An AVIF with its picture and an Exif item whose bytes are in the file, located by an iloc box (method 0)."""
    body = struct.pack(">I", 0) + blob  # the offset to the TIFF header, then the blob
    iinf = full_box(b"iinf", 0, struct.pack(">H", 2) + infe(1, b"av01") + infe(2, b"Exif"))
    hdlr, pitm = full_box(b"hdlr", 0, b"\x00" * 21), full_box(b"pitm", 0, b"\x00\x01")

    def build(offset: int) -> bytes:
        entries = (
            struct.pack(">HHHH", 1, 0, 0, 1)
            + struct.pack(">II", 0, 4)
            + struct.pack(">HHHH", 2, 0, 0, 1)
            + struct.pack(">II", offset, len(body))
        )
        # item_ID(2) data_reference(2) extent_count(2) is not what a version 0 box holds: item_ID, data_reference_index, extent_count
        entries = (
            struct.pack(">HHH", 1, 0, 1)
            + struct.pack(">II", 0, 4)
            + struct.pack(">HHH", 2, 0, 1)
            + struct.pack(">II", offset, len(body))
        )
        iloc = full_box(b"iloc", 0, bytes([0x44, 0x00]) + struct.pack(">H", 2) + entries)
        meta = full_box(b"meta", 0, hdlr + pitm + iloc + iinf)
        return box(b"ftyp", b"avif\x00\x00\x00\x00avifmif1") + meta

    start = len(build(0))
    return build(start) + body


if __name__ == "__main__":
    unittest.main()
