import { packRows } from "./lanes.js";

// Pure logic of the Review page: which items need a look, the timeline geometry, hit testing and the
// conversion of a drag on the picture into a box in source pixels.

export function needsCheck(item) {
  return item.judgement.verdict === "check" && !item.reviewed;
}

// Every item the detectors were not sure about, decided or not: what the two bulk buttons act on.
export function doubtfulItems(items) {
  return items.filter((item) => item.judgement.verdict === "check");
}

export function foundBySecondLook(item) {
  return item.found_by === "second";
}

// The bulk decision for the doubtful items: nothing is sent unless the person confirms. `ask(kind, count)` resolves true on
// confirm; `send(blur)` makes the request. Returns whether it was sent.
export async function bulkDecision({ blur, count, ask, send }) {
  if (count === 0) return false;
  if (!(await ask(blur ? "blur" : "keep", count))) return false;
  await send(blur);
  return true;
}

export function visibleItems(items, onlyCheck, onlySecond = false) {
  return items.filter((item) => (!onlyCheck || needsCheck(item)) && (!onlySecond || foundBySecondLook(item)));
}

const byFirstFrame = (a, b) => a.first_frame - b.first_frame || a.item_id - b.item_id;

// The next (or previous) item that still needs a decision, wrapping around; null when none does.
export function nextToCheck(items, frame, direction = 1) {
  const pending = items.filter(needsCheck).sort(byFirstFrame);
  if (pending.length === 0) return null;
  if (direction > 0) return pending.find((item) => item.first_frame > frame) ?? pending[0];
  return [...pending].reverse().find((item) => item.first_frame < frame) ?? pending[pending.length - 1];
}

// The item after (or before) the selected one in time order, wrapping around: how the keyboard reaches any item, for a video has no list of rows
// and its bars take a pointer. With nothing selected it is the first after (or the last before) `frame`.
export function neighbourItem(items, selectedId, frame, direction = 1) {
  const ordered = [...items].sort(byFirstFrame);
  if (ordered.length === 0) return null;
  const at = ordered.findIndex((item) => item.item_id === selectedId);
  if (at >= 0) return ordered[(at + direction + ordered.length) % ordered.length];
  if (direction > 0) return ordered.find((item) => item.first_frame > frame) ?? ordered[0];
  return [...ordered].reverse().find((item) => item.first_frame < frame) ?? ordered[ordered.length - 1];
}

export function activeAt(items, frame) {
  return items.filter((item) => item.first_frame <= frame && frame <= item.last_frame);
}

export function clampFrame(frame, frames) {
  return Math.min(Math.max(0, Math.round(frame)), Math.max(0, frames - 1));
}

export function frameAtX(x, width, frames) {
  if (width <= 0 || frames <= 1) return 0;
  return clampFrame((x / width) * (frames - 1), frames);
}

export function xAtFrame(frame, width, frames) {
  return frames <= 1 ? 0 : (frame / (frames - 1)) * width;
}

// One row per overlapping item, faces above plates, and the areas the person added in a lane of their own below (only when there
// are some); every bar is at least `minWidth` px wide so it can be clicked.
export function layoutTimeline(items, frames, width, { rowHeight = 18, laneGap = 10, minWidth = 6 } = {}) {
  const { placed, rows } = packRows(items);
  const faceRows = Math.max(rows.face, 1);
  const plateRows = Math.max(rows.plate, 1);
  const plateTop = faceRows * rowHeight + laneGap;
  const manualTop = plateTop + plateRows * rowHeight + laneGap;
  const tops = { face: 0, plate: plateTop, manual: manualTop };
  const rects = placed.map(({ item, lane, row }) => {
    const x = xAtFrame(item.first_frame, width, frames);
    const end = frames <= 1 ? width : xAtFrame(item.last_frame + 1, width, frames);
    return { item, lane, x, w: Math.min(Math.max(end - x, minWidth), Math.max(width - x, minWidth)), y: tops[lane] + row * rowHeight, h: rowHeight };
  });
  const lanes = { face: { top: 0, height: faceRows * rowHeight }, plate: { top: plateTop, height: plateRows * rowHeight } };
  if (rows.manual > 0) lanes.manual = { top: manualTop, height: rows.manual * rowHeight };
  const last = lanes.manual ?? lanes.plate;
  return { rects, rowHeight, height: last.top + last.height, lanes };
}

// The number a person reads for an item: its place in time order among all the items of the clip, 1 to n. The engine's own ids
// are stable names for the decisions and skip the candidates it dropped, so "Area #22" could stand in a list of twelve.
export function itemNumbers(items) {
  const ordered = [...items].sort((a, b) => a.first_frame - b.first_frame || a.item_id - b.item_id);
  return new Map(ordered.map((item, index) => [item.item_id, index + 1]));
}

// The bar under the pointer (a few px of slack sideways); with several, the one whose centre is nearest.
export function hitTimeline(layout, x, y, slack = 3) {
  let best = null;
  let bestDistance = Infinity;
  for (const rect of layout.rects) {
    if (y < rect.y || y >= rect.y + rect.h || x < rect.x - slack || x > rect.x + rect.w + slack) continue;
    const distance = Math.abs(x - (rect.x + rect.w / 2));
    if (distance < bestDistance) {
      best = rect;
      bestDistance = distance;
    }
  }
  return best ? best.item : null;
}

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

export function rulerTicks(durationS, width, minGap = 72) {
  if (!(durationS > 0) || width <= 0) return [];
  const step = TICK_STEPS.find((s) => (s / durationS) * width >= minGap) ?? TICK_STEPS[TICK_STEPS.length - 1];
  const ticks = [];
  for (let seconds = 0; seconds <= durationS; seconds += step) ticks.push({ seconds, x: (seconds / durationS) * width });
  return ticks;
}

// A drag between two client points over the picture, as [x0, y0, x1, y1] in source pixels; null when it is a click.
export function boxFromDrag(a, b, rect, size, minSide = 6) {
  if (rect.width <= 0 || rect.height <= 0) return null;
  const toSource = (point) => [
    Math.min(Math.max(((point.x - rect.left) / rect.width) * size.width, 0), size.width),
    Math.min(Math.max(((point.y - rect.top) / rect.height) * size.height, 0), size.height),
  ];
  const [ax, ay] = toSource(a);
  const [bx, by] = toSource(b);
  const box = [Math.round(Math.min(ax, bx)), Math.round(Math.min(ay, by)), Math.round(Math.max(ax, bx)), Math.round(Math.max(ay, by))];
  return box[2] - box[0] < minSide || box[3] - box[1] < minSide ? null : box;
}

// Frames of a missed area that starts at `frame` and lasts `seconds` (null: to the end of the clip).
export function areaFrames(frame, seconds, fps, frames) {
  const last = Math.max(0, frames - 1);
  if (frames <= 1) return { frame_from: 0, frame_to: 0 };
  const to = seconds == null ? last : Math.min(last, frame + Math.max(1, Math.round(seconds * fps)) - 1);
  return { frame_from: Math.min(frame, last), frame_to: to };
}

export function itemKindKey(item) {
  return `review.item.${item.kind}`;
}

export function certaintyPercent(item) {
  return Math.round((item.max_score ?? 0) * 100);
}

// A plate band is a guess from where a plate usually sits: the engine gives it the constant 1.0, which is not a certainty and is not shown as one.
export const POSITION_BAND = "plate.reason.position_band";
export function isPositionBand(item) {
  return item.judgement?.reason === POSITION_BAND;
}

export function certaintyKey(item) {
  if (item.manual) return "review.cert.manual";
  if (isPositionBand(item)) return needsCheck(item) ? "review.cert.band_check" : item.reviewed ? "review.cert.band_decided" : "review.cert.band";
  if (needsCheck(item)) return "review.cert.check";
  return item.reviewed ? "review.cert.decided" : "review.cert.sure";
}

// Mock mode only: the boxes of the drawn scene that belong to the items active at a frame.
export function sceneBoxes(items, frame, detections) {
  return activeAt(items, frame).map((item) => {
    const same = detections.filter((d) => d.kind === (item.kind === "face" ? "face" : "plate"));
    const box = item.box ?? same[item.item_id % same.length].box;
    return { item, box, blur: item.blur, check: item.judgement.verdict === "check" };
  });
}
