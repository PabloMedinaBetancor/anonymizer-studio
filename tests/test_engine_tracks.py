"""E5: tracks, gaps and margins. The synthetic gap round (cameras,
objects and a noisy detector with misses) and the 1.04 checks P1-P10, A3, A4."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from anonymizer.engine import tracks as tk
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from anonymizer.contract import FACE, PLATE, RETRY, SWEEP, Detection
from tests.fakes import V1_SHAPE, V1_VIDEO_MARGINS, v1_video_judgement, v1_wants_recover, v1_wants_verify

W, H, FPS = 1080, 1920, 30.0
SPECS = {kind: tk.BlurSpec(V1_VIDEO_MARGINS[kind], V1_SHAPE[kind]) for kind in (FACE, PLATE)}


def camera(n, rng, kind):
    """Similarities from t-1 to t of a handheld camera: still, panning or walking (zoom, changing pan, shake)."""
    motions = [tk.IDENTITY.copy()]
    for t in range(1, n):
        if kind == "still":
            dx, dy, angle, zoom = rng.normal(0, 0.3), rng.normal(0, 0.3), 0, 1
        elif kind == "pan":
            dx, dy, angle, zoom = (
                6 + rng.normal(0, 1.5),
                rng.normal(0, 1.5),
                rng.normal(0, 0.002),
                1 + rng.normal(0, 0.001),
            )
        else:
            dx, dy = 8 * np.sin(t / 25) + rng.normal(0, 3), 3 * np.cos(t / 17) + rng.normal(0, 3)
            angle, zoom = rng.normal(0, 0.004), 1.004 + rng.normal(0, 0.002)
        c, s = np.cos(angle) * zoom, np.sin(angle) * zoom
        cx, cy = W / 2, H / 2
        motions.append(np.array([[c, -s, cx - c * cx + s * cy + dx], [s, c, cy - s * cx - c * cy + dy]]))
    return motions


def objects(n, motions, rng):
    """True paths: plates still in the scene (they follow the camera), faces with their own motion."""
    out = []
    for k in range(6):
        kind = PLATE if k < 4 else FACE
        w = rng.uniform(50, 160) if kind == PLATE else rng.uniform(40, 120)
        e = np.array([rng.uniform(200, W - 200), rng.uniform(300, H - 300), w, w / 4.2 if kind == PLATE else w * 1.25])
        v = np.zeros(2) if kind == PLATE else rng.normal(0, 2.0, 2)
        start, end = int(rng.integers(0, n // 3)), int(rng.integers(2 * n // 3, n))
        path = {}
        for t in range(n):
            if t > 0:
                e = tk.carry(motions[t], e)
                e[:2] += v
            if start <= t <= end and 0 < e[0] < W and 0 < e[1] < H:
                path[t] = e.copy()
        out.append((kind, path))
    return out


def detections(n, objs, rng, step, p_seen, streak, noise):
    """What the detector sees at each sweep: isolated misses (1 - p_seen), two streaks of misses per object, position
    noise, and a weak false plate now and then. As (box, score, kind, on_vehicle)."""
    by_t = {t: [] for t in range(n)}
    off = {}
    for k in range(len(objs)):
        off[k] = set()
        for t0 in rng.choice(n, size=2, replace=False):
            off[k].update(range(int(t0), int(t0) + streak * step))
    for t in range(0, n, step):
        for k, (kind, path) in enumerate(objs):
            if t not in path or t in off[k] or rng.random() > p_seen:
                continue
            e = path[t].copy()
            e[:2] += rng.normal(0, noise * e[2], 2)
            e[2:] *= 1 + rng.normal(0, noise, 2)
            score = float(np.clip(rng.normal(0.85 if kind == FACE else 0.7, 0.08), 0.3, 1))
            by_t[t].append((tk.state_to_box(e), score, kind, kind == PLATE))
        if rng.random() < 0.05:
            e = np.array([rng.uniform(100, W - 100), rng.uniform(100, H - 100), 60, 20])
            by_t[t].append((tk.state_to_box(e), 0.3, PLATE, False))
    return by_t


def coverage(truth, box):
    iw = max(0, min(truth[2], box[2]) - max(truth[0], box[0]))
    ih = max(0, min(truth[3], box[3]) - max(truth[1], box[1]))
    return iw * ih / max(1e-9, (truth[2] - truth[0]) * (truth[3] - truth[1]))


class Round:
    """One synthetic clip run through this engine, with retries and verifications simulated from the truth."""

    def __init__(self, seed, camera_kind, step, p_seen=0.8, streak=4, noise=0.04, n=300):
        rng = np.random.default_rng(seed)
        self.n, self.step = n, step
        self.motions = camera(n, rng, camera_kind)
        self.objects = objects(n, self.motions, rng)
        self.by_t = detections(n, self.objects, rng, step, p_seen, streak, noise)

    def _truth_near(self, kind, t, box):
        best = None
        for kk, path in self.objects:
            if kk == kind and t in path:
                b = tk.state_to_box(path[t])
                if best is None or coverage(b, box) > coverage(best, box):
                    best = b
        return best if best is not None and coverage(best, box) > 0.2 else None

    def run(self):
        mine = tk.Tracker(FPS, tk.TrackerParams())
        for t in range(self.n):
            key = t % self.step == 0
            raw = self.by_t[t] if key else []
            lost = mine.step(t, self.motions[t], False, [Detection(*d[0], d[1], d[2], SWEEP, d[3]) for d in raw], key)
            for pm in lost:
                if not v1_wants_recover(pm.stats()):
                    continue
                truth = self._truth_near(pm.kind, t, tk.state_to_box(pm.state))
                if truth is not None:
                    mine.recover(pm, t, Detection(*truth, 0.6, pm.kind, RETRY, pm.kind == PLATE))
            if key:
                for pm in mine.active:
                    dm = pm.views.get(t)
                    if dm is None or dm.source != SWEEP:
                        continue
                    if v1_wants_verify(pm.stats(), dm):
                        v = 0.95 if dm.score > 0.35 else 0.0
                        pm.best_verify = max(pm.best_verify, v)
                        pm.n_verify += 1
                        pm.verify_side = max(dm.w, dm.h)
            mine.end_step(t, FPS / self.step)
        self.tracks = tk.join_split(
            tk.merge_duplicates(mine.finish()), mine.motions, mine.cuts, self.n, FPS, tk.TrackerParams()
        )
        self.mine = mine
        return self


class SyntheticRound(unittest.TestCase):
    def test_E5_motion_fills_gaps_at_least_as_well_as_linear_and_hold(self):
        # % of object-frames covered whole, 3 seeds:
        # pan/4 motion 97.0, linear 96.3, hold 96.5; walk/3 97.5, 96.5, 95.1; walk/4 97.1, 95.5, 92.1. With the 1.04
        # margins a steady pan leaves linear and hold level, so linear >= hold is required of the handheld camera.
        for cam, step in (("pan", 4), ("walk", 3), ("walk", 4)):
            covered = {m: [0, 0] for m in (tk.MOTION, tk.LINEAR, tk.HOLD)}
            for seed in range(3):
                r = Round(seed, cam, step, p_seen=0.75, streak=5).run()
                blurred = [p for p in r.tracks if v1_video_judgement(p.stats()).verdict.value == "blur"]
                for m in covered:
                    boxes = tk.final_boxes(
                        blurred, r.mine.motions, r.mine.cuts, r.n, FPS, tk.TrackerParams(), W, H, SPECS, m
                    )
                    for _, path in r.objects:
                        for t, e in path.items():
                            truth = tk.state_to_box(e)
                            seen = (max(0, truth[0]), max(0, truth[1]), min(W, truth[2]), min(H, truth[3]))
                            if (seen[2] - seen[0]) * (seen[3] - seen[1]) < 0.3 * e[2] * e[3]:
                                continue
                            covered[m][1] += 1
                            if max((coverage(seen, b) for b, _, _ in boxes.get(t, [])), default=0) >= 0.99:
                                covered[m][0] += 1
            rate = {m: c / n for m, (c, n) in covered.items()}
            self.assertGreaterEqual(rate[tk.MOTION], rate[tk.LINEAR], f"{cam} {step}: {rate}")
            self.assertGreaterEqual(rate[tk.MOTION], rate[tk.HOLD], f"{cam} {step}: {rate}")
            if cam == "walk":
                self.assertGreaterEqual(rate[tk.LINEAR], rate[tk.HOLD], f"{cam} {step}: {rate}")


def shift(dx=0.0, dy=0.0, zoom=1.0):
    return np.array([[zoom, 0, dx], [0, zoom, dy]], float)


def run(n, motions, dets_by_t, keys=None, cuts=()):
    tracker = tk.Tracker(FPS)
    for t in range(n):
        key = True if keys is None else t in keys
        tracker.step(t, motions[t], t in cuts, dets_by_t.get(t, []), key)
        tracker.end_step(t, FPS)
    return tracker, tracker.finish()


def plate(box, score=0.8, source=SWEEP):
    return Detection(*box, score, PLATE, source, True)


class GapRules(unittest.TestCase):
    def setUp(self):
        self.params = tk.TrackerParams()

    def test_E5_seen_missed_seen_is_one_track_and_the_gap_follows_the_camera(self):
        motions = [shift()] + [shift(10, 0)] * 9
        truth, e = {}, tk.box_to_state((100, 100, 180, 120))
        for t in range(10):
            if t:
                e = tk.carry(motions[t], e)
            truth[t] = e.copy()
        tracker, found = run(10, motions, {t: [plate(tk.state_to_box(truth[t]))] for t in (0, 2, 3, 4, 9)})
        self.assertEqual(len(found), 1)
        filled = tk.complete(found[0], tracker.motions, tracker.cuts, 10, self.params, FPS)
        self.assertEqual(filled[1][1], "gap")
        self.assertLess(float(np.abs(filled[1][0] - truth[1]).max()), 1)
        self.assertLess(max(float(np.abs(filled[t][0] - truth[t]).max()) for t in range(5, 9)), 1)
        held = tk.complete(found[0], tracker.motions, tracker.cuts, 10, self.params, FPS, tk.HOLD)
        self.assertGreaterEqual(float(np.abs(held[6][0] - truth[6]).max()), 19)

    def test_E5_extended_0_3_s_each_side_but_never_across_a_cut(self):
        dets = {t: [plate((200, 200, 260, 214))] for t in range(20, 30)}
        tracker, found = run(40, [shift()] * 40, dets)
        filled = tk.complete(found[0], tracker.motions, tracker.cuts, 40, self.params, FPS)
        self.assertEqual((min(filled), max(filled)), (11, 38))
        tracker, found = run(40, [shift()] * 40, dets, cuts={18})
        filled = tk.complete(found[0], tracker.motions, tracker.cuts, 40, self.params, FPS)
        self.assertEqual(min(filled), 18)

    def test_E5_two_tracks_on_one_plate_merge_and_the_neighbour_does_not(self):
        a, b, c = tk.Track(0, PLATE), tk.Track(1, PLATE), tk.Track(2, PLATE)
        for t in range(10):
            a.views[t] = plate((100, 100, 140, 110))
        for t in range(4, 14):
            b.views[t] = Detection(101, 100, 141, 111, 0.6, PLATE, RETRY, True)
            c.views[t] = plate((400, 100, 440, 110), 0.7)
        b.on_vehicle = True
        merged = tk.merge_duplicates([a, b, c])
        self.assertEqual([p.id for p in merged], [0, 2])
        self.assertEqual(sorted(merged[0].views), list(range(14)))
        self.assertTrue(merged[0].on_vehicle)

    def test_E5_a_runaway_scale_cannot_grow_the_patch_beyond_its_cap(self):
        motions = [shift()] + [shift(zoom=1.2)] * 29
        dets = {0: [plate((100, 100, 140, 110))], 29: [plate((100, 100, 140, 110))]}
        tracker, found = run(30, motions, dets)
        boxes = tk.final_boxes(found, tracker.motions, tracker.cuts, 30, FPS, self.params, 4000, 4000, SPECS)
        widths = [b[2] - b[0] for frame in boxes.values() for b, _, _ in frame]
        # 1.5 times what was seen (40 px), with the largest margin (far extension: 0.35 + 0.20 per side)
        self.assertLessEqual(max(widths), 40 * 1.5 * 2.1 + 1)

    def test_E5_the_final_box_always_contains_the_detected_box(self):
        r = Round(5, "walk", 2).run()
        boxes = tk.final_boxes(r.tracks, r.mine.motions, r.mine.cuts, r.n, FPS, self.params, W, H, SPECS)
        for track in r.tracks:
            for t, d in track.views.items():
                mine = [b for b, _, pid in boxes.get(t, []) if pid == track.id]
                x0, y0 = max(0.0, d.x0), max(0.0, d.y0)
                x1, y1 = min(float(W), d.x1), min(float(H), d.y1)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                self.assertTrue(mine, f"track {track.id} frame {t}")
                b = mine[0]
                self.assertTrue(b[0] <= x0 and b[1] <= y0 and b[2] >= x1 and b[3] >= y1, f"{b} vs {d.box()}")

    def test_E5_minimum_margin_in_pixels_scales_with_the_video(self):
        tracker, found = run(5, [shift()] * 5, {t: [plate((500, 500, 520, 506))] for t in range(5)})
        small = tk.final_boxes(found, tracker.motions, tracker.cuts, 5, FPS, self.params, 1920, 1080, SPECS)[2][0][0]
        large = tk.final_boxes(found, tracker.motions, tracker.cuts, 5, FPS, self.params, 3840, 2160, SPECS)[2][0][0]
        self.assertAlmostEqual(500 - small[0], 3.0)
        self.assertAlmostEqual(500 - large[0], 6.0)

    def test_E5_unknown_gap_method_is_refused(self):
        tracker, found = run(3, [shift()] * 3, {0: [plate((0, 0, 40, 10))]})
        with self.assertRaises(ValueError):
            tk.complete(found[0], tracker.motions, tracker.cuts, 3, self.params, FPS, "spline")


class SplitTracks(unittest.TestCase):
    """Checks A3 and A4."""

    def follow(self, n, seen, cuts=()):
        tracker = tk.Tracker(FPS)
        for t in range(n):
            cut = t in cuts
            key = t % 2 == 0 or cut
            dets = [Detection(100, 100, 148, 148, 0.99, FACE)] if key and seen(t) else []
            tracker.step(t, tk.IDENTITY.copy(), cut, dets, key)
            tracker.end_step(t, FPS / 2)
        found = tk.merge_duplicates(tracker.finish())
        joined = tk.join_split(found, tracker.motions, tracker.cuts, n, FPS, tk.TrackerParams())
        boxes = tk.final_boxes(joined, tracker.motions, tracker.cuts, n, FPS, tk.TrackerParams(), 640, 480, SPECS)
        return joined, [t for t in range(n) if t not in boxes]

    def test_E5_a_face_lost_for_1_4_s_in_the_same_place_stays_one_track_and_covered(self):
        joined, uncovered = self.follow(120, lambda t: not 30 <= t < 72)
        self.assertEqual((len(joined), uncovered), (1, []))

    def test_E5_a_one_frame_flash_does_not_cut(self):
        _, uncovered = self.follow(90, lambda t: t != 45, cuts={45, 46})
        self.assertEqual(uncovered, [])
        self.assertEqual(tk.effective_cuts({45, 46}, 90), set())
        self.assertEqual(tk.effective_cuts({45}, 90), {45})
        self.assertEqual(tk.effective_cuts({30, 88}, 90), {30})

    def test_E5_faces_far_apart_or_across_a_real_cut_are_not_joined(self):
        counts = []
        for far, cut_at in ((True, None), (False, 40)):
            tracker = tk.Tracker(FPS)
            for t in range(90):
                if t < 20:
                    d = Detection(100, 100, 148, 148, 0.99, FACE)
                elif t >= 62:
                    d = Detection(*((400, 300, 448, 348) if far else (100, 100, 148, 148)), 0.99, FACE)
                else:
                    d = None
                key = t % 2 == 0
                tracker.step(
                    t, tk.IDENTITY.copy(), t == cut_at, [d] if d is not None and key else [], key or t == cut_at
                )
                tracker.end_step(t, FPS / 2)
            found = tk.merge_duplicates(tracker.finish())
            counts.append(len(tk.join_split(found, tracker.motions, tracker.cuts, 90, FPS, tk.TrackerParams())))
        self.assertEqual(counts, [2, 2])


if __name__ == "__main__":
    unittest.main()
