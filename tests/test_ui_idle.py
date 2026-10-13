"""The Studio no longer stops itself after 60 seconds (QA and tablet reports, 04/10/2026): a tablet that locks its screen or a browser
that freezes a background tab came back to a dead page. It waits 10 minutes with nothing loaded and 6 hours while clips (their
analyses and the person's decisions, held in memory only) are loaded, and says so."""

from __future__ import annotations

import time
import unittest

try:
    from anonymizer.ui import server as server_module
    from tests import test_ui_server as server_tests
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class IdleStop(server_tests.StudioCase):
    def gone_for(self, seconds):
        self.server.last_request = time.monotonic() - seconds

    def test_a_page_gone_for_a_minute_does_not_stop_the_server(self):
        self.gone_for(61)
        self.assertFalse(self.server.idle_expired())
        self.gone_for(90)
        self.assertFalse(self.server.idle_expired())

    def test_with_nothing_loaded_it_stops_after_ten_minutes(self):
        self.gone_for(server_module.IDLE_SECONDS - 5)
        self.assertFalse(self.server.idle_expired())
        self.gone_for(server_module.IDLE_SECONDS + 5)
        self.assertTrue(self.server.idle_expired())
        self.assertEqual(server_module.IDLE_SECONDS, 600)

    def test_with_clips_loaded_it_waits_six_hours(self):
        self.import_clip()
        self.gone_for(3 * 3600)
        self.assertFalse(self.server.idle_expired())
        self.gone_for(6 * 3600 + 5)
        self.assertTrue(self.server.idle_expired())

    def test_the_watchdog_stops_a_server_that_has_waited_long_enough(self):
        self.gone_for(10)
        server_module.watch_idle(self.server, seconds=5)
        self.thread.join(20)
        self.assertFalse(self.thread.is_alive())

    def test_the_page_is_told_the_waits(self):
        state = self.api("GET", "/api/state")
        self.assertEqual(state["idle_stop"], {"seconds": 600, "seconds_with_clips": 21600})


if __name__ == "__main__":
    unittest.main()
