"""The breakdown of the missed faces (docs/MISSES.md) describes the published figures: the per-face log it is computed from reproduces the totals of the sealed read
exactly, and the tables are arithmetic on that log and nothing else."""

from __future__ import annotations

import json
import os
import unittest

try:
    from bench import miss_breakdown as mb
except ImportError as e:  # the bench libraries (OpenCV, numpy) are not installed
    raise unittest.SkipTest(f"bench libraries missing: {e}") from e

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "results")


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def read_text(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def face(
    part: str,
    height: float,
    careful: bool,
    fast: bool = False,
    deep: bool = False,
    photo_h: int = 1000,
    wider=None,
    best=None,
) -> dict:
    def cell(hidden: bool) -> dict:
        return {"hidden": hidden, "certain": hidden, "ellipse": hidden, "touching": 1 if best else 0, "best": best}

    out = {
        "id": f"{part}/x.jpg#{height}",
        "part": part,
        "photo": f"{part}/x.jpg",
        "box": [0, 0, height, height],
        "photo_w": 1000,
        "photo_h": photo_h,
        "height_px": height,
        "width_px": height,
        "height_frac": round(height / photo_h, 5),
        "modes": {"fast": cell(fast), "careful": cell(careful), "deep": cell(deep)},
    }
    if wider is not None:
        out["wider"] = wider
    return out


class SealedTotals(unittest.TestCase):
    def test_the_log_reproduces_the_totals_of_the_sealed_read(self):
        """Each mode and part, the faces and the faces hidden by default, plus the shares hidden with certainty and as ellipses of the read's cells."""
        log = load("miss-breakdown-log.json")
        self.assertEqual(log["keys"], "sealed")
        check = mb.check_totals(log, load("sealed-v2.json"))
        wrong = [row for row in check["rows"] if not row["same"]]
        self.assertEqual(wrong, [])
        self.assertTrue(check["match"])

    def test_the_headline_totals_are_the_published_ones(self):
        log = load("miss-breakdown-log.json")
        independent = [f for f in log["faces"] if f["part"] in mb.INDEPENDENT]
        self.assertEqual(len(independent), 1222)
        self.assertEqual(sum(1 for f in independent if f["modes"]["careful"]["hidden"]), 935)
        self.assertEqual(sum(1 for f in independent if f["modes"]["fast"]["hidden"]), 866)
        self.assertEqual(sum(1 for f in independent if f["modes"]["deep"]["hidden"]), 996)

    def test_the_tables_add_up_to_the_totals_of_the_log(self):
        log = load("miss-breakdown-log.json")
        t = mb.tables(log, load("sealed-v2.json"))
        for name, cell in t["sets"].items():
            for key in ("by_height_px", "by_height_frac"):
                self.assertEqual(sum(r["faces"] for r in cell[key]), cell["faces"], f"{name} {key}")
                for m in t["modes"]:
                    self.assertEqual(
                        sum(r[m]["hidden"] for r in cell[key]), cell["overall"][m]["hidden"], f"{name} {key} {m}"
                    )
                    self.assertEqual(
                        sum(r[m]["missed"] for r in cell[key]), cell["overall"][m]["missed"], f"{name} {key} {m}"
                    )
        stored = load("miss-breakdown.json")
        self.assertEqual(json.loads(json.dumps(t, sort_keys=True))["sets"], {k: v for k, v in stored["sets"].items()})


class Writeups(unittest.TestCase):
    def test_every_missed_blind_face_has_a_tag(self):
        log, tags = load("miss-breakdown-log.json"), load("miss-blind-tags.json")
        missed = {
            f["id"]
            for f in log["faces"]
            if f["part"] == "blind" and not all(f["modes"][m]["hidden"] for m in mb.LOG_MODES)
        }
        self.assertEqual(missed, set(tags["faces"]))
        for entry in tags["faces"].values():
            self.assertIn(entry["primary"], mb.TAGS)
            self.assertIn(entry["primary"], entry["tags"])
            self.assertTrue(set(entry["tags"]) <= set(mb.TAGS))

    def test_the_tables_file_is_what_the_log_gives(self):
        log, sealed, tags = load("miss-breakdown-log.json"), load("sealed-v2.json"), load("miss-blind-tags.json")
        stored = load("miss-breakdown.json")
        fresh = json.loads(json.dumps(mb.tables(log, sealed, tags), sort_keys=True))
        stored.pop("log", None)
        self.assertEqual(fresh, stored)

    def test_a_miss_with_nothing_blurred_is_a_miss_no_face_zone_touched(self):
        """Two fields of the log, written apart, say the same thing: share 0 (the blurred zones cover none of the face) and no zone touching it."""
        t = load("miss-breakdown.json")
        for mode in t["modes"]:
            self.assertEqual(
                t["sets"]["independent"]["loose"][mode]["untouched"],
                t["sets"]["independent"]["causes"][mode]["not_found"],
                mode,
            )

    def test_the_written_page_is_what_the_tables_give(self):
        """docs/MISSES.md carries no hand-written figure: its generated blocks are the tables' own."""
        path = os.path.join(ROOT, "docs", "MISSES.md")
        self.assertTrue(
            mb.write_docs(path, mb.render_blocks(load("miss-breakdown.json")), check=True),
            "run: python3 -s -m bench.miss_breakdown docs",
        )

    def test_the_page_has_every_block_it_generates(self):
        text = read_text("docs", "MISSES.md")
        for name in mb.render_blocks(load("miss-breakdown.json")):
            self.assertIn(mb.BEGIN.format(name), text, name)
            self.assertIn(mb.END.format(name), text, name)

    def test_the_readme_numbers_are_markers_into_the_headline(self):
        text = read_text("docs", "accuracy.template.md")
        self.assertIn("Why some faces are missed", text)
        self.assertIn("docs/MISSES.md", text)
        head = load("miss-breakdown.json")["misses_headline"]
        for marker in (
            "careful.misses_under_20_px_pct",
            "careful.misses_under_40_px_pct",
            "careful.hidden_at_least_40_px_pct",
        ):
            self.assertIn("{misses_headline." + marker + "}", text)
            self.assertIsNotNone(head["careful"][marker.split(".", 1)[1]])


class Arithmetic(unittest.TestCase):
    def setUp(self):
        self.log = {
            "modes": ["fast", "careful", "deep"],
            "faces": [
                face("wider", 8, False),
                face("wider", 15, False),
                face("wider", 15, True),
                face("wider", 30, True, fast=True),
                face("wider", 50, True, fast=True, deep=True),
                face(
                    "wider",
                    50,
                    False,
                    best={
                        "cover": 0.5,
                        "verdict": "check",
                        "found_by": "light",
                        "score": 0.4,
                        "second_score": None,
                        "blur": True,
                    },
                ),
                face("wider", 120, True, fast=True, deep=True),
            ],
        }

    def test_height_bins_partition_the_faces(self):
        rows = mb._bins(self.log["faces"], ["careful"], mb.HEIGHT_BINS_PX, "height_px")
        self.assertEqual([r["faces"] for r in rows], [1, 2, 1, 2, 1])
        self.assertEqual([r["careful"]["missed"] for r in rows], [1, 1, 0, 1, 0])
        self.assertAlmostEqual(sum(r["careful"]["share_of_all_misses_pct"] for r in rows), 100.0, places=1)

    def test_a_cut_says_how_many_misses_are_small_and_what_is_hidden_above_it(self):
        row = next(r for r in mb._cuts(self.log["faces"], ["careful"]) if r["cut_px"] == 40)
        self.assertEqual(row["careful"]["misses"], 3)
        self.assertEqual(row["careful"]["misses_under_cut"], 2)
        self.assertEqual(row["careful"]["hidden_at_least_cut"], 2)
        self.assertEqual(row["faces_at_least"], 3)

    def test_misses_are_split_into_not_found_and_found_but_covered_partly(self):
        causes = mb._causes(self.log["faces"], ["careful"])["careful"]
        self.assertEqual((causes["misses"], causes["not_found"], causes["found_but_covered_partly"]), (3, 2, 1))
        self.assertEqual(causes["found_but_covered_partly_by_verdict"], {"check": 1})
        self.assertEqual(
            causes["misses_unsure_and_not_hidden"], 0
        )  # unsure zones are blurred by default: the face was missed by cover, not by the verdict

    def test_the_check_refuses_a_log_that_differs_from_the_sealed_read(self):
        sealed = {
            "results": {
                "faces_v2": {
                    "parts": {
                        p: {m: {"hidden_by_default": 0, "faces": 0} for m in self.log["modes"]} for p in mb.PARTS
                    },
                    "independent_parts": {m: {"hidden_by_default": 0, "faces": 0} for m in self.log["modes"]},
                    "cells": {
                        m: {"blurred_with_certainty_pct": 0.0, "hidden_by_default_ellipse_pct": 0.0}
                        for m in self.log["modes"]
                    },
                }
            }
        }
        self.assertFalse(mb.check_totals(self.log, sealed)["match"])


class WiderAnnotationIsNotPublished(unittest.TestCase):
    """WIDER FACE is research-only data: the published log and tables carry no file name, no box and no picture size of its photos."""

    ALLOWED = {"id", "part", "photo", "height_px", "height_frac", "modes", "wider"}

    def test_the_wider_rows_of_the_log_have_opaque_ids_and_no_geometry(self):
        rows = [f for f in load("miss-breakdown-log.json")["faces"] if f["part"] == "wider"]
        self.assertEqual(len(rows), 1144)
        for f in rows:
            self.assertLessEqual(set(f), self.ALLOWED)
            self.assertRegex(f["id"], r"^wider#\d{4}$")
            self.assertRegex(f["photo"], r"^wider-photo-\d{3}$")
        self.assertEqual(len({f["id"] for f in rows}), len(rows))
        self.assertEqual(len({f["photo"] for f in rows}), 86)

    def test_no_published_result_names_a_wider_photo(self):
        import re

        for name in ("miss-breakdown-log.json", "miss-breakdown.json"):
            with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
                text = f.read()
            self.assertIsNone(re.search(r"wider/[^\"]*\.jpg", text), name)
            self.assertIsNone(re.search(r"\d+--[A-Za-z_]+__", text), name)
        for entry in load("miss-breakdown.json")["wider_careful_misses_at_least_40px"]:
            self.assertEqual(set(entry), {"id", "height_px", "hindrance", "best"})
            self.assertRegex(entry["id"], r"^wider#\d{4}$")

    def test_the_rows_do_not_depend_on_the_order_of_the_annotation(self):
        faces = []
        for i, h in enumerate(range(10, 60)):
            attrs = {"blur": i % 3, "expression": 0, "illumination": 0, "occlusion": i % 3, "pose": 0}
            f = face("wider", float(h), careful=bool(i % 2), wider=attrs)
            f["photo"] = f"wider/p{i % 7}.jpg"
            f["id"] = f"{f['photo']}#{i}"
            faces.append(f)
        faces.append(face("blind", 30.0, True))
        out = mb.anonymise_wider(faces)
        self.assertEqual(out, mb.anonymise_wider(list(reversed(faces))))
        wider = [f for f in out if f["part"] == "wider"]
        self.assertEqual(len(wider), 50)
        self.assertEqual(len({f["photo"] for f in wider}), 7)
        for f in wider:
            self.assertLessEqual(set(f), self.ALLOWED)
        self.assertEqual([f for f in out if f["part"] == "blind"], [face("blind", 30.0, True)])


if __name__ == "__main__":
    unittest.main()
