"""What the video decoder of this machine does with a damaged clip, the thing three privacy tests (P02, P15, P16) depend on.

    python3 -s scripts/probe_damaged_decode.py

It makes the clip those tests make (60 frames, 400 bytes scrambled in the middle of the video data) and prints, for the PyAV and
FFmpeg of this machine, how many frames came out, whether the decoder raised, and how many frames it marked corrupt. The engine
reports a damaged file only when the decoder raises (engine/video_io.py); on a machine whose FFmpeg conceals the damage instead
(printed as "frames out" equal to the 60 written and no error), the tests fail with "UnsupportedMedia not raised" while the
engine behaves as designed. Run it on the aarch64 tablet to see which of the two it is.
"""

from __future__ import annotations

import os
import sys
import tempfile

import av

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import media  # noqa: E402
from tests.test_privacy_paths import damage  # noqa: E402


def main() -> int:
    print("av", av.__version__, "libavcodec", av.library_versions.get("libavcodec"), flush=True)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "clip.mp4")
        media.moving_plate_clip(path, n=60, audio=False)
        for label in ("intact", "damaged"):
            if label == "damaged":
                damage(path)
            frames = corrupt = 0
            error = None
            try:
                with av.open(path) as container:
                    for frame in container.decode(container.streams.video[0]):
                        frames += 1
                        corrupt += bool(getattr(frame, "is_corrupt", False))
            except Exception as e:  # what the engine's handler would see
                error = f"{type(e).__module__}.{type(e).__name__}: {e}"
            print(f"{label}: frames out {frames}, marked corrupt {corrupt}, decoder raised: {error}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
