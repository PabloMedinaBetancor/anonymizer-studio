"""Which zones the second look visits and how persons are found (criteria SL1, SL2)."""

from __future__ import annotations

import unittest

import numpy as np

from anonymizer.classes import context
from anonymizer.classes.plates import PlateModule
from anonymizer.contract import (
    FACE,
    PLATE,
    ZONE_DOUBTFUL,
    ZONE_DROPPED_STRONG,
    ZONE_DROPPED_WEAK,
    ZONE_PERSON_HEAD,
    ZONE_VEHICLE,
    Detection,
    Judgement,
    Verdict,
)
from anonymizer.engine import second_look, zones
from anonymizer.engine.zones import LookItem, build_zones


def item(box, verdict, score=0.8, kind=FACE) -> LookItem:
    return LookItem(box, kind, score, verdict, Judgement(verdict, "face.reason.test"))


def person(box, score=0.9) -> Detection:
    return Detection(*box, score, context.PERSON)


def vehicle(box) -> Detection:
    return Detection(*box, 0.9, "vehicle")


class BuildZones(unittest.TestCase):
    def test_sl1_the_zones_come_in_the_priority_order_of_the_spec(self):
        items = [
            item((10, 10, 40, 40), Verdict.DROP, 0.55),  # weak: 4
            item((500, 10, 540, 50), Verdict.CHECK),  # doubtful: 1
            item((300, 300, 330, 330), Verdict.DROP, 0.7),  # strong: 2
            item((900, 900, 930, 930), Verdict.BLUR),  # sure: never visited
        ]
        found = build_zones(items, [person((100, 400, 200, 700))], [vehicle((600, 600, 900, 800))])
        self.assertEqual(
            [(z.why, z.kind, z.item) for z in found],
            [
                (ZONE_DOUBTFUL, FACE, 1),
                (ZONE_DROPPED_STRONG, FACE, 2),
                (ZONE_PERSON_HEAD, FACE, None),
                (ZONE_DROPPED_WEAK, FACE, 0),
                (ZONE_VEHICLE, PLATE, None),
            ],
        )
        self.assertEqual([z.priority for z in found], [0, 1, 2, 3, 4])

    def test_sl1_a_person_with_a_face_zone_has_no_head_zone(self):
        p = person((100, 100, 300, 700))
        for verdict in (Verdict.BLUR, Verdict.CHECK):
            found = build_zones([item((160, 120, 220, 190), verdict)], [p])
            self.assertEqual([z.why for z in found if z.why == ZONE_PERSON_HEAD], [], verdict)

    def test_sl1_a_face_in_the_lower_half_or_a_dropped_candidate_does_not_count_as_the_persons_face(self):
        p = person((100, 100, 300, 700))
        lower = build_zones([item((160, 500, 220, 560), Verdict.BLUR)], [p])
        self.assertEqual([z.why for z in lower], [ZONE_PERSON_HEAD])
        dropped = build_zones([item((160, 120, 220, 190), Verdict.DROP, 0.55)], [p])
        self.assertIn(ZONE_PERSON_HEAD, [z.why for z in dropped])

    def test_sl1_the_head_zone_is_the_top_third_of_the_person_widened_a_tenth(self):
        (z,) = build_zones([], [person((100, 200, 200, 600))])
        self.assertEqual(z.box, (90.0, 200.0, 210.0, 340.0))

    def test_sl1_two_zones_on_the_same_place_are_one_visit(self):
        items = [item((100, 100, 140, 140), Verdict.CHECK), item((104, 102, 142, 141), Verdict.DROP, 0.7)]
        found = build_zones(items)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].why, ZONE_DOUBTFUL)

    def test_sl1_a_small_zone_inside_a_much_bigger_one_is_still_visited(self):
        items = [item((100, 100, 130, 130), Verdict.DROP, 0.55)]
        found = build_zones(items, [person((0, 90, 400, 900))])  # head zone about 480 x 280 px around it
        self.assertEqual([z.why for z in found], [ZONE_PERSON_HEAD, ZONE_DROPPED_WEAK])

    def test_sl1_inside_a_priority_the_larger_zone_goes_first(self):
        items = [item((0, 0, 20, 20), Verdict.CHECK), item((100, 0, 160, 60), Verdict.CHECK)]
        found = build_zones(items)
        self.assertEqual([z.item for z in found], [1, 0])

    def test_sl1_vehicles_go_largest_first_and_classes_without_a_heavy_look_get_no_zones(self):
        found = build_zones(
            [item((0, 0, 20, 20), Verdict.CHECK, kind=PLATE)],
            [person((0, 0, 100, 300))],
            [vehicle((0, 0, 100, 100)), vehicle((200, 0, 600, 300))],
            kinds=(FACE,),
        )
        self.assertEqual([z.why for z in found], [ZONE_PERSON_HEAD])
        found = build_zones([], [], [vehicle((0, 0, 100, 100)), vehicle((200, 0, 600, 300))])
        self.assertEqual([z.box for z in found], [(200, 0, 600, 300), (0, 0, 100, 100)])

    def test_sl1_the_zone_list_leaves_the_items_untouched(self):
        items = [item((100, 100, 140, 140), Verdict.CHECK)]
        before = [(i.box, i.verdict) for i in items]
        build_zones(items, [person((0, 0, 300, 600))])
        self.assertEqual([(i.box, i.verdict) for i in items], before)


class RecordingVehicles:
    """A vehicle (or person) detector that records the regions it is asked about and returns the same box in every view."""

    def __init__(self, kind):
        self.kind, self.regions = kind, []

    def detect(self, bgr, region=None, min_score=None):
        self.regions.append(region)
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = region or (0, 0, w, h)
        return [Detection(x0 + 10, y0 + 10, x0 + 90, y0 + 150, 0.9, self.kind)]


class PersonPass(unittest.TestCase):
    def _vehicle_regions(self, bgr):
        module = object.__new__(PlateModule)
        module.timings, module.calls = {}, {}
        module.vehicles = RecordingVehicles("vehicle")
        module._find_vehicles(bgr, True)
        return module.vehicles.regions

    def test_sl2_the_person_pass_looks_where_the_vehicle_pass_looks(self):
        for shape in ((2000, 3000, 3), (1200, 1000, 3), (1300, 1500, 3)):
            bgr = np.zeros(shape, np.uint8)
            detector = RecordingVehicles(context.PERSON)
            context.find_persons(detector, bgr)
            self.assertEqual(detector.regions, self._vehicle_regions(bgr), shape)

    def test_sl2_repeats_in_overlapping_tiles_are_merged_as_the_vehicle_pass_merges_them(self):
        class Same:
            def detect(self, bgr, region=None, min_score=None):
                return [Detection(1000, 900, 1100, 1100, 0.9, context.PERSON)]

        found = context.find_persons(Same(), np.zeros((2000, 3000, 3), np.uint8))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kind, context.PERSON)

    def test_sl2_a_person_inside_a_bigger_person_box_is_kept_by_the_merge(self):
        # same class of fault as the vehicle merge (B, 03/10/2026): the containment rule of the face `nms` dropped it
        class Nested:
            def detect(self, bgr, region=None, min_score=None):
                return [
                    Detection(100, 100, 700, 900, 0.9, context.PERSON),
                    Detection(300, 300, 400, 420, 0.8, context.PERSON),
                ]

        found = context.find_persons(Nested(), np.zeros((2000, 3000, 3), np.uint8))
        self.assertEqual(sorted(d.box() for d in found), [(100, 100, 700, 900), (300, 300, 400, 420)])

    def test_sl2_a_child_in_front_of_an_adult_gets_a_head_zone_of_its_own(self):
        # RV-RFDETR F1: B's case (the child's box lies inside the adult's); the whole path, from the finder to the zones
        class Family:
            def detect(self, bgr, region=None, min_score=None):
                return [
                    Detection(100, 100, 300, 600, 0.9, context.PERSON),
                    Detection(150, 300, 250, 600, 0.8, context.PERSON),
                ]

        persons = context.find_persons(Family(), np.zeros((2000, 3000, 3), np.uint8))
        found = build_zones([], persons)
        self.assertEqual(len(found), 2)
        child_head = context.head_zone((150, 300, 250, 600))
        self.assertTrue(
            any(z.box == child_head and z.why == zones.ZONE_PERSON_HEAD for z in found), [z.box for z in found]
        )

    def test_sl2_persons_are_context_and_never_items_of_the_second_look(self):
        result = second_look.second_stage(
            np.zeros((1000, 1000, 3), np.uint8), [], [], persons=[person((0, 0, 200, 600))]
        )
        self.assertEqual(result.items, [])
        self.assertEqual(result.zones_total, 0)
        self.assertNotIn(context.PERSON, [z.kind for z in zones.build_zones([], [person((0, 0, 200, 600))])])

    def test_sl2_the_person_detector_takes_only_class_zero_and_labels_it_person(self):
        class Session:
            def get_inputs(self):
                class Input:
                    name, shape = "images", [1, 3, 416, 416]

                return [Input()]

        detector = context.PersonDetector(Session())
        self.assertEqual(list(detector.classes), [0])
        self.assertEqual(detector.min_score, context.PERSON_SCORE)


if __name__ == "__main__":
    unittest.main()
