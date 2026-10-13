import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { performance } from "perf_hooks";
import {
  needsCheck, nextToCheck, neighbourItem, activeAt, layoutTimeline, hitTimeline, frameAtX, xAtFrame, rulerTicks, boxFromDrag, areaFrames, certaintyKey, itemNumbers,
} from "../../src/anonymizer/ui/static/js/review-logic.js";

const item = (id, kind, first, last, verdict = "blur", reviewed = false) => ({ item_id: id, kind, first_frame: first, last_frame: last, judgement: { verdict }, reviewed, blur: verdict === "blur", manual: false, max_score: 0.5 });

test("only unreviewed check items need a look", () => {
  assert.equal(needsCheck(item(1, "face", 0, 5, "check")), true);
  assert.equal(needsCheck(item(2, "face", 0, 5, "check", true)), false);
  assert.equal(needsCheck(item(3, "face", 0, 5, "blur")), false);
});

test("N goes to the next item to check after the frame and wraps", () => {
  const items = [item(1, "face", 10, 20, "check"), item(2, "plate", 30, 40, "check"), item(3, "face", 50, 60)];
  assert.equal(nextToCheck(items, 0).item_id, 1);
  assert.equal(nextToCheck(items, 10).item_id, 2);
  assert.equal(nextToCheck(items, 30).item_id, 1);
  assert.equal(nextToCheck(items, 35, -1).item_id, 2);
  assert.equal(nextToCheck(items, 5, -1).item_id, 2);
  assert.equal(nextToCheck([item(3, "face", 0, 1)], 0), null);
});

test("activeAt lists the items visible at a frame, ends included", () => {
  const items = [item(1, "face", 10, 20), item(2, "plate", 20, 30)];
  assert.deepEqual(activeAt(items, 20).map((i) => i.item_id), [1, 2]);
  assert.deepEqual(activeAt(items, 21).map((i) => i.item_id), [2]);
});

test("frame and x convert both ways and clamp", () => {
  assert.equal(frameAtX(500, 1000, 101), 50);
  assert.equal(frameAtX(-20, 1000, 101), 0);
  assert.equal(frameAtX(5000, 1000, 101), 100);
  assert.equal(xAtFrame(100, 1000, 101), 1000);
  assert.equal(frameAtX(10, 1000, 1), 0);
});

test("timeline: faces above plates, rows do not overlap and every bar can be clicked", () => {
  const layout = layoutTimeline([item(1, "face", 0, 1000), item(2, "face", 500, 600), item(3, "plate", 0, 2), item(4, "plate", 990, 991)], 1000, 1000);
  const rect = (id) => layout.rects.find((r) => r.item.item_id === id);
  assert.equal(rect(1).y, 0);
  assert.equal(rect(2).y, layout.rowHeight);
  assert.ok(rect(3).y > rect(2).y);
  for (const r of layout.rects) {
    assert.ok(r.w >= 6);
    assert.ok(r.x + r.w <= 1000 + 6);
  }
  assert.equal(layout.height, layout.lanes.plate.top + layout.lanes.plate.height);
});

test("hit test picks the bar under the pointer and nothing on empty space", () => {
  const layout = layoutTimeline([item(1, "face", 0, 100), item(2, "face", 50, 150)], 1000, 1000);
  assert.equal(hitTimeline(layout, 30, 5).item_id, 1);
  assert.equal(hitTimeline(layout, 100, layout.rowHeight + 5).item_id, 2);
  assert.equal(hitTimeline(layout, 500, 5), null);
  assert.equal(hitTimeline(layout, 30, 400), null);
});

test("500 items: layout and 1000 hit tests stay fast and every bar selects itself (S3)", () => {
  const items = [];
  for (let i = 0; i < 500; i += 1) {
    const first = (i * 37) % 9000;
    items.push(item(i, i % 3 === 0 ? "plate" : "face", first, first + 20 + ((i * 13) % 300), i % 7 === 0 ? "check" : "blur"));
  }
  const started = performance.now();
  const layout = layoutTimeline(items, 10000, 1400);
  for (let i = 0; i < 1000; i += 1) hitTimeline(layout, (i * 7) % 1400, (i * 5) % layout.height);
  assert.ok(performance.now() - started < 400);
  for (const r of layout.rects) {
    const centre = hitTimeline(layout, r.x + r.w / 2, r.y + r.h / 2);
    assert.equal(centre.item_id, r.item.item_id);
  }
  assert.equal(new Set(layout.rects.map((r) => r.item.item_id)).size, 500);
});

test("ruler ticks are spaced at least the minimum gap apart", () => {
  const ticks = rulerTicks(108, 1500);
  assert.ok(ticks.length > 3);
  for (let i = 1; i < ticks.length; i += 1) assert.ok(ticks[i].x - ticks[i - 1].x >= 72);
  assert.deepEqual(rulerTicks(0, 1000), []);
});

test("a drag becomes a clamped box in source pixels, a click becomes nothing", () => {
  const rect = { left: 100, top: 50, width: 960, height: 540 };
  const size = { width: 1920, height: 1080 };
  assert.deepEqual(boxFromDrag({ x: 300, y: 250 }, { x: 200, y: 150 }, rect, size), [200, 200, 400, 400]);
  assert.deepEqual(boxFromDrag({ x: 0, y: 0 }, { x: 5000, y: 5000 }, rect, size), [0, 0, 1920, 1080]);
  assert.equal(boxFromDrag({ x: 300, y: 250 }, { x: 301, y: 251 }, rect, size), null);
});

test("an added area lasts the chosen seconds, to the end, or one frame for a photo", () => {
  assert.deepEqual(areaFrames(100, 5, 25, 1000), { frame_from: 100, frame_to: 224 });
  assert.deepEqual(areaFrames(900, 10, 25, 1000), { frame_from: 900, frame_to: 999 });
  assert.deepEqual(areaFrames(100, null, 25, 1000), { frame_from: 100, frame_to: 999 });
  assert.deepEqual(areaFrames(0, 5, 0, 1), { frame_from: 0, frame_to: 0 });
});

test("certainty text depends on whether the user already decided", () => {
  assert.equal(certaintyKey(item(1, "face", 0, 1, "check")), "review.cert.check");
  assert.equal(certaintyKey(item(1, "face", 0, 1, "check", true)), "review.cert.decided");
  assert.equal(certaintyKey({ ...item(1, "face", 0, 1), manual: true }), "review.cert.manual");
});

test("timeline: the bars of a photo span the whole width so they can be clicked", () => {
  const layout = layoutTimeline([item(1, "face", 0, 0), item(2, "face", 0, 0)], 1, 1000);
  assert.equal(layout.rects[0].w, 1000);
  assert.equal(hitTimeline(layout, 500, 5).item_id, 1);
});

test("the bulk buttons act on every doubtful item, decided or not, and on nothing else", async () => {
  const { doubtfulItems } = await import("../../src/anonymizer/ui/static/js/review-logic.js");
  const item = (id, verdict, reviewed) => ({ item_id: id, judgement: { verdict }, reviewed });
  assert.deepEqual(doubtfulItems([item(1, "blur", false), item(2, "check", false), item(3, "check", true)]).map((i) => i.item_id), [2, 3]);
  assert.deepEqual(doubtfulItems([item(1, "blur", false)]), []);
});

test("the filters combine: only items to check, only second-look finds, both", async () => {
  const { visibleItems, foundBySecondLook } = await import("../../src/anonymizer/ui/static/js/review-logic.js");
  const item = (id, verdict, reviewed, found_by) => ({ item_id: id, judgement: { verdict }, reviewed, found_by });
  const items = [item(1, "blur", false, "light"), item(2, "check", false, "second"), item(3, "check", true, "second"), item(4, "check", false, "light")];
  assert.deepEqual(visibleItems(items, false, false).map((i) => i.item_id), [1, 2, 3, 4]);
  assert.deepEqual(visibleItems(items, true, false).map((i) => i.item_id), [2, 4]);
  assert.deepEqual(visibleItems(items, false, true).map((i) => i.item_id), [2, 3]);
  assert.deepEqual(visibleItems(items, true, true).map((i) => i.item_id), [2]);
  assert.equal(foundBySecondLook(items[1]), true);
});

test("the bulk decision sends nothing unless the person confirms, and sends the boolean once when they do", async () => {
  const { bulkDecision } = await import("../../src/anonymizer/ui/static/js/review-logic.js");
  const sent = [];
  const send = async (blur) => sent.push(blur);
  assert.equal(await bulkDecision({ blur: false, count: 3, ask: async () => false, send }), false);
  assert.deepEqual(sent, []);
  assert.equal(await bulkDecision({ blur: false, count: 3, ask: async () => true, send }), true);
  assert.deepEqual(sent, [false]);
  const asked = [];
  await bulkDecision({ blur: true, count: 2, ask: async (kind, count) => { asked.push([kind, count]); return true; }, send });
  assert.deepEqual(asked, [["blur", 2]]);
  assert.deepEqual(sent, [false, true]);
  assert.equal(await bulkDecision({ blur: true, count: 0, ask: async () => { throw new Error("nothing to ask"); }, send }), false);
});

test("timeline: an added area is drawn in a lane of its own below the plates, and only when there is one", () => {
  const layout = layoutTimeline([item(1, "face", 0, 10), item(2, "plate", 0, 10), item(30, "manual", 0, 10)], 1000, 1000);
  const manual = layout.rects.find((r) => r.item.item_id === 30);
  assert.equal(manual.lane, "manual");
  assert.ok(manual.y >= layout.lanes.plate.top + layout.lanes.plate.height);
  assert.equal(layout.height, layout.lanes.manual.top + layout.lanes.manual.height);
  assert.equal(layoutTimeline([item(1, "face", 0, 10)], 1000, 1000).lanes.manual, undefined);
});

test("items are numbered 1 to n in time order whatever the engine's ids are (a dropped candidate leaves a gap in them)", () => {
  const numbers = itemNumbers([item(22, "manual", 40, 90), item(7, "plate", 10, 20), item(3, "face", 10, 20), item(15, "face", 0, 5)]);
  assert.deepEqual([...numbers.entries()], [[15, 1], [3, 2], [7, 3], [22, 4]]);
  assert.equal(numbers.size, 4);
});

test("the keyboard walks every item in time order, wraps, and starts from the frame when nothing is selected", () => {
  const items = [item(7, "plate", 50, 60), item(3, "face", 10, 20, "check"), item(5, "face", 30, 40)];
  assert.equal(neighbourItem(items, 3, 0, 1).item_id, 5);
  assert.equal(neighbourItem(items, 5, 0, 1).item_id, 7);
  assert.equal(neighbourItem(items, 7, 0, 1).item_id, 3);
  assert.equal(neighbourItem(items, 3, 0, -1).item_id, 7);
  assert.equal(neighbourItem(items, 5, 0, -1).item_id, 3);
  assert.equal(neighbourItem(items, null, 25, 1).item_id, 5);
  assert.equal(neighbourItem(items, null, 25, -1).item_id, 3);
  assert.equal(neighbourItem(items, null, 99, 1).item_id, 3);
  assert.equal(neighbourItem([], null, 0, 1), null);
});
