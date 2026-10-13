"""What the Studio server needs from the API to draw Review: media_info, thumbnails, boxes and preview frames."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

try:
    import cv2
    from PIL import Image

    import anonymizer
    from anonymizer.contract import FACE, PLATE
    from anonymizer.engine import video_io
    from tests import media
    from tests.fakes import FixedModule, plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class PreviewCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)


class VideoPreview(PreviewCase):
    def setUp(self):
        super().setUp()
        self.clip = self.path("clip.mp4")
        media.moving_plate_clip(self.clip, audio=False)
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            self.analysis = anon.analyze(self.clip)
        [self.item] = self.analysis.items

    def test_read_frame_matches_sequential_decoding_and_ends_with_none(self):
        for t in (0, 1, 17, 59):
            np.testing.assert_array_equal(video_io.read_frame(self.clip, t), media.decoded_frame(self.clip, t))
        self.assertIsNone(video_io.read_frame(self.clip, 60))

    def test_boxes_follow_the_plate_and_carry_the_decision(self):
        found = self.analysis.boxes(30, 30)[30]
        self.assertEqual(
            [(b.item_id, b.kind, b.shape, b.blur) for b in found], [(self.item.item_id, PLATE, "rect", True)]
        )
        want = media.plate_box(30)
        box = found[0].box
        self.assertLessEqual(box[0], want[0])
        self.assertGreaterEqual(box[2], want[2])
        self.analysis.set_blur(self.item.item_id, False)
        self.assertEqual(self.analysis.boxes(30, 30)[30], [])
        shown = self.analysis.boxes(30, 30, only_blurred=False)[30]
        self.assertEqual([b.blur for b in shown], [False])

    def test_extra_margin_widens_the_boxes(self):
        plain = self.analysis.boxes(30, 30)[30][0].box
        wide = self.analysis.boxes(30, 30, anonymizer.Options(extra_margin=0.2))[30][0].box
        self.assertLess(wide[0], plain[0])
        self.assertGreater(wide[2], plain[2])

    def test_anonymized_view_hides_the_plate_and_original_does_not(self):
        x0, y0, x1, y1 = (int(v) for v in media.plate_box(30))
        inside = (slice(y0 + 6, y1 - 6), slice(x0 + 6, x1 - 6))
        original = decode(self.analysis.preview(30, "original", 960))
        hidden = decode(self.analysis.preview(30, "anonymized", 960))
        self.assertEqual(original.shape, (240, 320, 3))
        self.assertLess(hidden[inside].std(), 0.4 * original[inside].std())
        np.testing.assert_array_equal(hidden[:60], original[:60])

    def test_a_decision_turned_off_shows_the_plate_again(self):
        self.analysis.set_blur(self.item.item_id, False)
        np.testing.assert_array_equal(
            decode(self.analysis.preview(30, "anonymized")), decode(self.analysis.preview(30, "original"))
        )

    def test_split_and_boxes_views(self):
        original = decode(self.analysis.preview(30, "original"))
        split = decode(self.analysis.preview(30, "split"))
        self.assertEqual(split.shape, original.shape)
        self.assertLess(np.abs(split[:, :150].astype(int) - original[:, :150]).mean(), 3)
        self.assertGreater(np.abs(split[:, 170:].astype(int) - original[:, 170:]).mean(), 0.5)
        self.assertGreater(np.abs(decode(self.analysis.preview(30, "boxes")).astype(int) - original).sum(), 0)

    def test_width_scales_down_and_never_up(self):
        self.assertEqual(decode(self.analysis.preview(0, "original", 200)).shape, (150, 200, 3))
        self.assertEqual(decode(self.analysis.preview(0, "original", 5000)).shape[1], 320)

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            self.analysis.preview(0, "sepia")
        with self.assertRaises(ValueError):
            self.analysis.preview(60, "original")
        with self.assertRaises(ValueError):
            self.analysis.boxes(5, 61)

    def test_a_changed_source_is_not_previewed(self):
        with open(self.clip, "ab") as f:
            f.write(b"\0")
        with self.assertRaises(anonymizer.SourceChanged):
            self.analysis.preview(0, "original")

    def test_added_areas_appear_in_boxes_and_stay_after_a_reload(self):
        area = self.analysis.add_area(10, 20, (5, 5, 45, 25), follow="fixed")
        self.assertEqual([b.item_id for b in self.analysis.boxes(15, 15)[15]], [self.item.item_id, area.item_id])
        self.assertEqual([b.item_id for b in self.analysis.boxes(25, 25)[25]], [self.item.item_id])
        saved = self.path("a.json")
        self.analysis.save(saved)
        again = anonymizer.load_analysis(saved)
        self.assertEqual(again.boxes(15, 15), self.analysis.boxes(15, 15))
        self.analysis.remove_area(area.item_id)
        self.assertEqual([b.item_id for b in self.analysis.boxes(15, 15)[15]], [self.item.item_id])

    def test_thumbnails_are_saved_next_to_the_analysis_and_come_back(self):
        saved = self.path("with-thumbs.analysis.json")
        self.analysis.save(saved)
        folder = self.path("with-thumbs.analysis.thumbs")
        self.assertEqual(os.listdir(folder), [f"{self.item.item_id}.jpg"])
        again = anonymizer.load_analysis(saved)
        self.assertEqual(again.thumbnail_jpeg(self.item.item_id), self.analysis.thumbnail_jpeg(self.item.item_id))
        self.assertEqual(again.thumbnail(self.item.item_id).shape, self.analysis.thumbnail(self.item.item_id).shape)
        area = again.add_area(5, 9, (5, 5, 45, 25))
        self.assertIsNone(again.thumbnail_jpeg(area.item_id))
        elsewhere = self.path("copy.json")
        again.save(elsewhere)
        self.assertEqual(os.listdir(self.path("copy.thumbs")), [f"{self.item.item_id}.jpg"])

    def test_item_thumbnail_is_a_jpeg_and_params_are_listed(self):
        self.assertEqual(self.analysis.thumbnail_jpeg(self.item.item_id)[:3], b"\xff\xd8\xff")
        self.assertIsInstance(self.item.params, dict)
        self.assertIsNone(self.analysis.thumbnail_jpeg(999))


class PhotoPreview(PreviewCase):
    PLATE_BOX, FACE_BOX = (20, 20, 100, 44), (140, 60, 200, 120)

    def setUp(self):
        super().setUp()
        self.noise = (np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)
        self.photo = self.path("f.png")
        Image.fromarray(self.noise).save(self.photo)
        modules = [FixedModule(FACE, [self.FACE_BOX], 0.85, 0.75, placed=True), FixedModule(PLATE, [self.PLATE_BOX])]
        with anonymizer.Anonymizer(modules=modules) as anon:
            self.analysis = anon.analyze(self.photo)

    def test_boxes_use_the_item_ids_and_shapes(self):
        found = self.analysis.boxes(0, 0, only_blurred=False)[0]
        by_id = {b.item_id: b for b in found}
        self.assertEqual({i.item_id for i in self.analysis.items}, set(by_id))
        face = next(i for i in self.analysis.items if i.kind == FACE)
        self.assertEqual((by_id[face.item_id].shape, by_id[face.item_id].blur), ("ellipse", True))  # doubtful: blurred
        self.assertEqual([b.kind for b in self.analysis.boxes(0, 0)[0]], [FACE, PLATE])
        self.analysis.set_blur(face.item_id, False)  # a person turned it off
        self.assertEqual([b.kind for b in self.analysis.boxes(0, 0)[0]], [PLATE])

    def test_preview_blurs_what_the_render_would(self):
        x0, y0, x1, y1 = self.PLATE_BOX
        shown = decode(self.analysis.preview(0, "anonymized"))
        original = self.noise[:, :, ::-1]
        self.assertLess(
            shown[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].std(), 0.5 * original[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].std()
        )

    def test_media_info_and_thumbnail(self):
        info = anonymizer.media_info(self.photo)
        self.assertEqual(
            (info.media, info.width, info.height, info.frames, info.has_gps), ("photo", 240, 160, 1, False)
        )
        self.assertEqual(info.size_bytes, os.path.getsize(self.photo))
        self.assertEqual(decode(anonymizer.thumbnail_jpeg(self.photo, 200)).shape, (133, 200, 3))

    def test_media_info_of_a_video_and_of_unsupported_files(self):
        clip = self.path("c.mp4")
        media.moving_plate_clip(clip, n=30)
        info = anonymizer.media_info(clip)
        self.assertEqual(
            (info.media, info.width, info.height, info.frames, info.has_audio), ("video", 320, 240, 30, True)
        )
        self.assertAlmostEqual(info.duration_s, 1.0, places=1)
        self.assertEqual(anonymizer.thumbnail_jpeg(clip)[:3], b"\xff\xd8\xff")
        for name in ("notes.txt", "photo.heic"):
            with open(self.path(name), "wb") as f:
                f.write(b"x" * 20)
            with self.assertRaises(anonymizer.UnsupportedMedia):
                anonymizer.media_info(self.path(name))
        with open(self.path("broken.jpg"), "wb") as f:
            f.write(b"not a jpeg")
        with self.assertRaises(anonymizer.UnsupportedMedia):
            anonymizer.media_info(self.path("broken.jpg"))


if __name__ == "__main__":
    unittest.main()
