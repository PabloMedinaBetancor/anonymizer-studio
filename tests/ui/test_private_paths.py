"""The capture gate of the real Studio: a page that shows a home folder or a user name is never captured."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from real_studio import STAGE, private_hits  # noqa: E402
except ImportError as e:  # real_studio drives Chrome through websocket-client, an optional test dependency
    raise unittest.SkipTest(f"the capture gate needs websocket-client: {e}") from e

HOME = "/ho" + "me/"
MAC = "/Us" + "ers/"
BACKSLASH = chr(92)


class PrivatePaths(unittest.TestCase):
    def test_finds_home_and_work_folders(self):
        self.assertEqual(private_hits("Save to " + HOME + "ana/Documents/x"), [HOME])
        self.assertEqual(
            private_hits("C:" + BACKSLASH + "Users" + BACKSLASH + "x"), []
        )  # Windows paths are not what this machine can show
        self.assertEqual(private_hits(MAC + "ana/Videos"), [MAC])

    def test_neutral_destination_is_clean(self):
        self.assertEqual(private_hits(str(STAGE / "anonymized")), [])


if __name__ == "__main__":
    unittest.main()
