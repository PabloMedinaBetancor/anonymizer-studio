"""The Export screen's size is a range that holds what the engine really wrote (QA report of 04/10/2026: one figure, 2.5 times too low on
the videos). The measured rows are the output of scripts/measure_export_size.py on the four demo files, run on the second laptop on 04/10/2026."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from anonymizer.ui import export_size

# (name, media, width, height, frames, duration_s, has_audio, size_bytes, bytes written)
MEASURED = (
    ("street-hongkong.mp4", "video", 960, 540, 300, 10.0, True, 347305, 764492),
    ("street-madrid.mp4", "video", 960, 540, 360, 15.0, True, 734691, 1894753),
    ("street-taipei.jpg", "photo", 1280, 540, 1, 0.0, False, 142040, 185769),
    ("street-westminster.jpg", "photo", 1280, 540, 1, 0.0, False, 114977, 160890),
)


def info(row):
    _, media, width, height, frames, duration, audio, size, _ = row
    return SimpleNamespace(
        media=media, width=width, height=height, frames=frames, duration_s=duration, has_audio=audio, size_bytes=size
    )


class Estimate(unittest.TestCase):
    def test_the_range_holds_every_measured_output(self):
        for row in MEASURED:
            low, high = export_size.estimate(info(row))
            self.assertLessEqual(low, row[-1], row[0])
            self.assertGreaterEqual(high, row[-1], row[0])

    def test_it_is_a_range_not_a_point_and_a_video_does_not_follow_its_original(self):
        small, large = info(MEASURED[0]), info(MEASURED[0])
        large.size_bytes = 100 * small.size_bytes  # a high-bitrate original of the same picture
        self.assertEqual(export_size.estimate(small), export_size.estimate(large))
        low, high = export_size.estimate(small)
        self.assertGreater(high, 2 * low)

    def test_the_old_estimate_the_size_of_the_original_missed_the_videos(self):
        for row in MEASURED[:2]:
            self.assertGreater(row[-1], 2 * row[7], row[0])  # what the tester saw: 2.2 and 2.6 times the original


if __name__ == "__main__":
    unittest.main()
