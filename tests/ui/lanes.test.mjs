import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { performance } from "perf_hooks";
import { packRows, isNotSure } from "../../src/anonymizer/ui/static/js/lanes.js";

const item = (id, kind, first, last, verdict = "blur") => ({ item_id: id, kind, first_frame: first, last_frame: last, judgement: { verdict } });

test("overlapping items of one kind take separate rows", () => {
  const { placed, rows } = packRows([item("a", "face", 0, 100), item("b", "face", 50, 150), item("c", "face", 101, 200)]);
  const rowOf = Object.fromEntries(placed.map((p) => [p.item.item_id, p.row]));
  assert.deepEqual(rowOf, { a: 0, b: 1, c: 0 });
  assert.equal(rows.face, 2);
});

test("faces and plates are packed independently", () => {
  const { rows } = packRows([item("a", "face", 0, 100), item("b", "plate", 0, 100)]);
  assert.deepEqual(rows, { face: 1, plate: 1, manual: 0 });
});

test("an item that starts the frame after another ends reuses its row", () => {
  const { rows } = packRows([item("a", "plate", 0, 10), item("b", "plate", 11, 20)]);
  assert.equal(rows.plate, 1);
});

test("an area the person added has its own lane, apart from the plates", () => {
  const { placed, rows } = packRows([item("m", "manual", 0, 5), item("p", "plate", 0, 5)]);
  assert.deepEqual(placed.map((p) => [p.item.item_id, p.lane]).sort(), [["m", "manual"], ["p", "plate"]]);
  assert.deepEqual(rows, { face: 0, plate: 1, manual: 1 });
});

test("packing 500 items finishes quickly and never overlaps within a row", () => {
  const items = Array.from({ length: 500 }, (_, i) => item(`i${i}`, i % 2 ? "face" : "plate", (i * 37) % 4000, ((i * 37) % 4000) + 30 + (i % 90)));
  const started = performance.now();
  const { placed } = packRows(items);
  assert.ok(performance.now() - started < 100);
  const byRow = new Map();
  for (const p of placed) {
    const key = `${p.lane}${p.row}`;
    const last = byRow.get(key);
    assert.ok(last === undefined || last < p.item.first_frame);
    byRow.set(key, p.item.last_frame);
  }
});

test("only check verdicts are not sure", () => {
  assert.equal(isNotSure(item("a", "face", 0, 1, "check")), true);
  assert.equal(isNotSure(item("a", "face", 0, 1, "blur")), false);
});
