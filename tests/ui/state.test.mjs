import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { initialState, reduce, itemCounts, estimateSeconds, overallFraction, analyzableClips, clipsToAnalyze, refusalNotices, projectName, remainingSeconds } from "../../src/anonymizer/ui/static/js/state.js";

const summary = (id, frames = 1000, extra = {}) => ({ clip_id: id, name: `${id}.mp4`, media: "video", source_dir: "/media/clips/Street shoot", duration_s: frames / 25, frames, width: 1920, height: 1080, fps: 25, has_audio: true, has_gps: false, status: "imported", analysis: null, ...extra });
const run = (messages, start = initialState()) => messages.reduce(reduce, start);
const added = (...ids) => ({ type: "clips.added", clips: ids.map((id) => summary(id)) });

test("added clips are imported, the first one is selected, repeats are ignored", () => {
  const state = run([added("a", "b"), added("a")]);
  assert.equal(state.clips.length, 2);
  assert.equal(state.selectedId, "a");
  assert.equal(state.clips[0].status, "imported");
});

test("loading the state restores clips analyzed in an earlier visit", () => {
  const analysis = { items: 4, to_check: 1, unchecked_frames: 0, rev: 2 };
  const state = run([{ type: "state.loaded", settings: { strength: 1.5 }, state: { mode: "local", languages: ["en"], capabilities: {}, demo_limits: null, job: null, clips: [summary("a", 1000, { status: "analyzed", analysis })] } }]);
  assert.equal(state.clips[0].status, "analyzed");
  assert.equal(state.clips[0].fraction, 1);
  assert.equal(state.options.strength, 1.5);
  assert.equal(state.info.mode, "local");
});

test("starting an analysis queues the listed clips and shows Analyze", () => {
  const state = run([added("a", "b"), { type: "job.started", job_id: "j", kind: "analyze", clips: [{ clip_id: "a" }] }]);
  assert.equal(state.tab, "analyze");
  assert.deepEqual(state.clips.map((c) => c.status), ["queued", "imported"]);
  assert.equal(state.job.state, "running");
});

test("a render job leaves the clips and the tab alone", () => {
  const state = run([added("a"), { type: "job.started", job_id: "j", kind: "render", clips: [{ clip_id: "a" }] }]);
  assert.equal(state.tab, "import");
  assert.equal(state.clips[0].status, "imported");
});

test("progress never goes backwards and keeps the live detections", () => {
  const live = [{ kind: "face", state: "found", score: 0.9, box: [1, 2, 3, 4] }];
  const state = run([
    added("a"),
    { type: "clip.progress", clip_id: "a", phase: "analyze", fraction: 0.6, frame: 600, eta_s: 10, live },
    { type: "clip.progress", clip_id: "a", phase: "analyze", fraction: 0.5, frame: 500, eta_s: 12 },
  ]);
  assert.equal(state.clips[0].fraction, 0.6);
  assert.equal(state.clips[0].status, "analyzing");
  assert.deepEqual(state.clips[0].live, []);
});

test("render progress does not touch the analysis state", () => {
  const before = run([added("a")]);
  const after = reduce(before, { type: "clip.progress", clip_id: "a", phase: "render", fraction: 0.2, frame: 1, eta_s: null });
  assert.equal(after.clips[0].fraction, before.clips[0].fraction);
  assert.deepEqual(after.clips[0].live, []);
  assert.equal(after.clips[0].render.fraction, 0.2);
});

test("an item announced twice is kept once", () => {
  const item = { item_id: 3, kind: "face", judgement: { verdict: "blur" }, first_frame: 0, last_frame: 9 };
  const state = run([added("a"), { type: "clip.item", clip_id: "a", item }, { type: "clip.item", clip_id: "a", item }]);
  assert.equal(state.clips[0].items.length, 1);
});

test("an analyzed clip records the counts and frames it could not check", () => {
  const state = run([added("a"), { type: "clip.analyzed", clip_id: "a", items: 3, to_check: 1, unchecked_frames: 12, rev: 0, analysis_url: "" }]);
  assert.equal(state.clips[0].status, "analyzed");
  assert.deepEqual(state.clips[0].analysis, { items: 3, to_check: 1, unchecked_frames: 12, rev: 0 });
});

test("a stopped job keeps analyzed clips and marks the rest stopped", () => {
  const state = run([
    added("a", "b", "c"),
    { type: "job.started", job_id: "j", kind: "analyze", clips: [{ clip_id: "a" }, { clip_id: "b" }, { clip_id: "c" }] },
    { type: "clip.analyzed", clip_id: "a", items: 0, to_check: 0, unchecked_frames: 0, rev: 0, analysis_url: "" },
    { type: "clip.progress", clip_id: "b", phase: "analyze", fraction: 0.3, frame: 300, eta_s: 5 },
    { type: "job.finished", job_id: "j", status: "stopped", summary: {} },
  ]);
  assert.deepEqual(state.clips.map((c) => c.status), ["analyzed", "stopped", "stopped"]);
  assert.equal(state.job, null);
  assert.deepEqual(analyzableClips(state).map((c) => c.id), ["a", "b", "c"]);
});

test("pause is the job state the server reports", () => {
  const state = run([{ type: "state.loaded", settings: {}, state: { mode: "local", languages: [], capabilities: {}, job: { job_id: "j", kind: "analyze", state: "running" }, clips: [] } }, { type: "job.state", state: "paused" }]);
  assert.equal(state.job.state, "paused");
});

test("errors and warnings become dismissible notices", () => {
  let state = run([{ type: "job.warning", code: "x.y", params: { n: 1 } }, { type: "job.error", code: "x.z" }]);
  assert.equal(state.notices.length, 2);
  state = reduce(state, { type: "ui.notice.dismiss", id: state.notices[0].id });
  assert.deepEqual(state.notices.map((n) => n.code), ["x.z"]);
});

test("item counts keep not-sure apart from faces and plates", () => {
  const mk = (kind, verdict) => ({ kind, judgement: { verdict } });
  assert.deepEqual(itemCounts([mk("face", "blur"), mk("face", "check"), mk("plate", "blur"), mk("manual", "blur")]), { faces: 1, plates: 2, not_sure: 1 });
});

test("without a measured speed there is no estimate, never a guess", () => {
  const clips = [{ frames: 1000 }, { frames: 500 }];
  assert.equal(estimateSeconds(clips, initialState()), null);
  const learned = reduce(initialState(), { type: "estimate", key: "standard/careful", seconds_per_frame: 0.02 });
  assert.equal(estimateSeconds(clips, learned), 30);
  assert.equal(estimateSeconds(clips, { ...learned, options: { ...learned.options, thoroughness: "maximum" } }), null);
});

test("time left for all clips adds the queued ones to the current estimate", () => {
  const state = run([added("a", "b"), { type: "estimate", key: "standard/careful", seconds_per_frame: 0.1 }, { type: "job.started", job_id: "j", kind: "analyze", clips: [{ clip_id: "a" }, { clip_id: "b" }] }]);
  const withCurrent = reduce(state, { type: "clip.started", clip_id: "a", phase: "analyze" });
  assert.equal(remainingSeconds(withCurrent, 20), 120);
  assert.equal(remainingSeconds(withCurrent, null), null);
});

test("overall fraction weighs clips by their frames", () => {
  const clips = [{ frames: 300, fraction: 1 }, { frames: 100, fraction: 0.5 }, { frames: 100, fraction: 0 }];
  assert.equal(overallFraction(clips), 0.7);
});

test("the project is named after the folder the clips came from", () => {
  assert.equal(projectName(run([added("a")]), "Untitled"), "Street shoot");
  assert.equal(projectName(initialState(), "Untitled"), "Untitled");
});

const itemOf = (id, first, extra = {}) => ({ item_id: id, kind: "plate", judgement: { verdict: "check", reason: "plate.reason.unconfirmed", params: {} }, first_frame: first, last_frame: first + 10, max_score: 0.4, blur: false, reviewed: false, manual: false, ...extra });
const analyzedState = () =>
  run([
    { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, demo_limits: null, job: null, clips: [summary("a", 1000, { status: "analyzed", analysis: { items: 2, to_check: 2, unchecked_frames: 0, rev: 1 } })] } },
    { type: "clip.items.loaded", clip_id: "a", items: [itemOf(1, 10), itemOf(2, 50)] },
  ]);

test("deciding an item replaces it and takes rev and to_check from the server", () => {
  const state = reduce(analyzedState(), { type: "item.changed", clip_id: "a", item: itemOf(1, 10, { blur: true, reviewed: true }), rev: 2, to_check: 1 });
  assert.equal(state.clips[0].items[0].blur, true);
  assert.deepEqual(state.clips[0].analysis, { items: 2, to_check: 1, unchecked_frames: 0, rev: 2 });
});

test("a bulk decision replaces every item it returns and takes rev and to_check from the server", () => {
  const changed = [itemOf(1, 10, { blur: true, reviewed: true }), itemOf(2, 50, { blur: true, reviewed: true })];
  const state = reduce(analyzedState(), { type: "items.changed", clip_id: "a", items: changed, rev: 3, to_check: 0 });
  assert.deepEqual(state.clips[0].items.map((i) => [i.item_id, i.blur, i.reviewed]), [[1, true, true], [2, true, true]]);
  assert.deepEqual(state.clips[0].analysis, { items: 2, to_check: 0, unchecked_frames: 0, rev: 3 });
});

test("a manual area is added in order of first frame and removed again with its selection", () => {
  let state = reduce(analyzedState(), { type: "area.added", clip_id: "a", item: itemOf(9, 30, { kind: "manual", manual: true, blur: true, reviewed: true }), rev: 2 });
  assert.deepEqual(state.clips[0].items.map((i) => i.item_id), [1, 9, 2]);
  assert.equal(state.clips[0].analysis.items, 3);
  state = reduce(state, { type: "review.select", item_id: 9 });
  state = reduce(state, { type: "area.removed", clip_id: "a", item_id: 9, rev: 3, to_check: 2 });
  assert.deepEqual(state.clips[0].items.map((i) => i.item_id), [1, 2]);
  assert.equal(state.review.itemId, null);
  assert.equal(state.clips[0].analysis.rev, 3);
});

test("opening Review selects the clip and clears the selected item; the review clip falls back to an analyzed one", async () => {
  const { reviewClip, isReady } = await import("../../src/anonymizer/ui/static/js/state.js");
  let state = reduce(analyzedState(), { type: "review.select", item_id: 2 });
  state = reduce(state, { type: "review.open", clip_id: "a" });
  assert.equal(state.tab, "review");
  assert.equal(state.review.itemId, null);
  assert.equal(reviewClip(state).id, "a");
  assert.equal(isReady(state.clips[0]), false);
  const ready = reduce(state, { type: "item.changed", clip_id: "a", item: itemOf(1, 10), rev: 2, to_check: 0 });
  assert.equal(isReady(ready.clips[0]), true);
});

test("the found counts of the real server are kept between progress events", () => {
  const state = run([
    added("a"),
    { type: "clip.progress", clip_id: "a", phase: "analyze", fraction: 0.3, frame: 300, eta_s: 10, found: { face: 3, plate: 1 } },
    { type: "clip.progress", clip_id: "a", phase: "analyze", fraction: 0.4, frame: 400, eta_s: 9 },
  ]);
  assert.deepEqual(state.clips[0].found, { face: 3, plate: 1 });
});

test("the cost of Careful is read from the protocol object", () => {
  const state = reduce(initialState(), { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, slowdown: { careful: 3.1 }, clips: [], job: null, seq: 1 } });
  assert.equal(state.info.costs.careful, 3.1);
});

test("the detection tier travels with the analysis, and the Studio has no Deep and no heavy-only", async () => {
  const { analyzeOptions, rateKey, DEFAULT_OPTIONS } = await import("../../src/anonymizer/ui/static/js/state.js");
  assert.equal(DEFAULT_OPTIONS.detection, "careful");
  assert.equal("heavy_only" in DEFAULT_OPTIONS, false);
  assert.deepEqual(analyzeOptions({ ...DEFAULT_OPTIONS, detection: "careful" }), { faces: true, plates: true, thoroughness: "standard", detection: "careful", gap_fill: "camera" });
  assert.deepEqual(analyzeOptions({ ...DEFAULT_OPTIONS, detection: "fast", heavy_only: true }), { faces: true, plates: true, thoroughness: "standard", detection: "fast", gap_fill: "camera" });
  assert.equal(rateKey({ thoroughness: "maximum", detection: "fast" }), "maximum/fast");
});

test("the second look's stage and zones are kept while a clip is analyzed and cleared when it is done", () => {
  let state = run([added("a")]);
  state = reduce(state, { type: "clip.progress", clip_id: "a", phase: "analyze", fraction: 0.9, frame: 900, eta_s: 3, stage: "second", zones: { done: 4, total: 12 } });
  assert.deepEqual([state.clips[0].stage, state.clips[0].zones], ["second", { done: 4, total: 12 }]);
  state = reduce(state, { type: "clip.progress", clip_id: "a", phase: "analyze", fraction: 0.2, frame: 200, eta_s: 3 });
  assert.deepEqual([state.clips[0].stage, state.clips[0].zones], ["first", null]);
  state = reduce(state, { type: "clip.analyzed", clip_id: "a", items: 1, to_check: 0, unchecked_frames: 0, rev: 1 });
  assert.deepEqual([state.clips[0].stage, state.clips[0].zones], [null, null]);
});

test("an error of the clip being analyzed makes it failed with its reason, not stopped; a render error leaves a clip analyzed", () => {
  const start = run([added("a", "b"), { type: "job.started", job_id: "j", kind: "analyze", clips: [{ clip_id: "a" }, { clip_id: "b" }] }, { type: "clip.started", clip_id: "a", phase: "analyze" }]);
  let state = reduce(start, { type: "job.error", code: "error.media.unreadable_photo", clip_id: "a", params: { name: "a.jpg" } });
  assert.equal(state.clips[0].status, "failed");
  assert.deepEqual(state.clips[0].failure, { code: "error.media.unreadable_photo", params: { name: "a.jpg" } });
  assert.equal(state.clips[1].status, "queued");
  state = reduce(state, { type: "clip.started", clip_id: "b", phase: "analyze" });
  state = reduce(state, { type: "job.finished", job_id: "j", status: "done" });
  assert.deepEqual(state.clips.map((c) => c.status), ["failed", "stopped"]);
  const rendering = run([added("a"), { type: "clip.analyzed", clip_id: "a", items: 0, to_check: 0, unchecked_frames: 0, rev: 1 }]);
  assert.equal(reduce(rendering, { type: "job.error", code: "x.y", clip_id: "a", params: {} }).clips[0].status, "analyzed");
});

test("the live badge and the frame count have no clock for a photo (its frame rate is 0)", async () => {
  const { liveClock, frameLabel } = await import("../../src/anonymizer/ui/static/js/format.js");
  const t = (key, params) => `${key} ${params.frame}/${params.total}`;
  assert.equal(liveClock({ media: "photo", fps: 0, frame: 0 }), "");
  assert.equal(frameLabel({ media: "photo", frames: 1, frame: 0 }, t), "");
  assert.equal(liveClock({ media: "video", fps: 25, frame: 125 }), "00:05");
  assert.equal(frameLabel({ media: "video", frames: 1000, frame: 12 }, t), "analyze.frame_of 12/1,000");
});

test("a size estimate is one figure when its ends are close and a range when they are not", async () => {
  const { sizeParts } = await import("../../src/anonymizer/ui/static/js/format.js");
  assert.deepEqual(sizeParts(900_000, 1_000_000), { key: "size", params: { size: "1.0 MB" } });
  assert.deepEqual(sizeParts(300_000, 1_200_000), { key: "range", params: { low: "300 KB", high: "1.2 MB" } });
  assert.equal(sizeParts(null, 5), null);
});

test("removing a clip from the list moves the selection to what is left, and to nothing when nothing is", () => {
  const summary = (id) => ({ clip_id: id, name: `${id}.jpg`, media: "photo", frames: 1, width: 10, height: 10, fps: 0, duration_s: 0, status: "imported" });
  let state = reduce(initialState(), { type: "clips.added", clips: [summary("c1"), summary("c2")] });
  state = reduce(state, { type: "ui.select", clip_id: "c1" });
  state = reduce(state, { type: "clip.removed", clip_id: "c1" });
  assert.deepEqual(state.clips.map((c) => c.id), ["c2"]);
  assert.equal(state.selectedId, "c2");
  state = reduce(state, { type: "clip.removed", clip_id: "c2" });
  assert.equal(state.selectedId, null);
  state = reduce(state, { type: "clips.added", clips: [summary("c3")] });
  assert.equal(state.selectedId, "c3");
});

test("a decision or an area that reaches this window twice (its answer and the event) is applied once; an older one is ignored", () => {
  const area = { type: "area.added", clip_id: "a", item: itemOf(9, 30, { kind: "manual", manual: true, blur: true, reviewed: true }), rev: 2 };
  let state = reduce(reduce(analyzedState(), area), area);
  assert.deepEqual(state.clips[0].items.map((i) => i.item_id), [1, 9, 2]);
  assert.equal(state.clips[0].analysis.items, 3);
  const removed = { type: "area.removed", clip_id: "a", item_id: 9, rev: 3, to_check: 2 };
  state = reduce(reduce(state, removed), removed);
  assert.equal(state.clips[0].analysis.items, 2);
  const stale = { type: "item.changed", clip_id: "a", item: { ...state.clips[0].items[0], blur: false }, rev: 1, to_check: 0 };
  assert.equal(reduce(state, stale), state);
});

test("a second window that has not loaded the items of a clip leaves the area to the load", () => {
  const unloaded = reduce(analyzedState(), { type: "clip.items.loaded", clip_id: "a", items: [] });
  const state = reduce(unloaded, { type: "area.added", clip_id: "a", item: itemOf(9, 30, { kind: "manual", manual: true }), rev: 2 });
  assert.deepEqual(state.clips[0].items, []);
});

test("a clip met in the middle of its analysis comes with its progress, what it found and, when it failed, why", () => {
  const loaded = (clips) => reduce(initialState(), { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, job: { job_id: "j1", kind: "analyze", state: "running" }, clips } });
  const item = { item_id: 0, kind: "face", judgement: { verdict: "blur" } };
  const running = loaded([{ ...summary("a", 1000), status: "analyzing", progress: { fraction: 0.4, frame: 400, eta_s: 30, found: { face: 1 }, stage: "first" }, items_so_far: [item] }]).clips[0];
  assert.equal(running.fraction, 0.4);
  assert.equal(running.eta_s, 30);
  assert.deepEqual(running.found, { face: 1 });
  assert.deepEqual(running.items, [item]);
  const failed = loaded([{ ...summary("b", 1000), status: "failed", failure: { code: "error.media.unreadable_video", params: { name: "b.mp4" } } }]).clips[0];
  assert.equal(failed.status, "failed");
  assert.deepEqual(failed.failure, { code: "error.media.unreadable_video", params: { name: "b.mp4" } });
});

test("what a window had on the Export screen comes back, each value checked", () => {
  const base = reduce(initialState(), { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, job: null, clips: [summary("a", 1000), summary("b", 1000)] } });
  const saved = { export: { options: { destination: "/x", name_pattern: "p_{index}", comparison: true, reports: { pdf: false, csv: "yes", json: true } }, excluded: [{ id: "a", name: "a.mp4" }, { id: "gone", name: "x.mp4" }, { id: "b", name: "another-file.mp4" }] } };
  const state = reduce(base, { type: "ui.restore", saved });
  assert.equal(state.export.options.destination, "/x");
  assert.equal(state.export.options.name_pattern, "p_{index}");
  assert.equal(state.export.options.comparison, true);
  assert.deepEqual(state.export.options.reports, { pdf: false, csv: true, json: true });  // "yes" is not a boolean: the default stays
  assert.deepEqual(state.export.excluded, ["a"]);  // a clip that is not there any more, or whose id now names another file, is dropped
  assert.equal(reduce(base, { type: "ui.restore", saved: "junk" }).export.options.name_pattern, "{name}_anonymized");
});

test("an analysed clip takes the frame count of its analysis, which a file's header can overstate by one", () => {
  const state = reduce(run([added("a")]), { type: "clip.analyzed", clip_id: "a", items: 1, to_check: 0, unchecked_frames: 0, rev: 1, frames: 999 });
  assert.equal(state.clips[0].frames, 999);
  const kept = reduce(run([added("a")]), { type: "clip.analyzed", clip_id: "a", items: 1, to_check: 0, unchecked_frames: 0, rev: 1 });
  assert.equal(kept.clips[0].frames, 1000);  // an event of an older server carries none
});

test("the same notice is not stacked twice, a different one is", () => {
  const error = { type: "job.error", code: "error.job.busy", params: {}, clip_id: null };
  const once = reduce(initialState(), error);
  const twice = reduce(once, error);
  assert.equal(twice.notices.length, 1);
  assert.equal(twice, once);  // nothing changed: nothing to draw again
  assert.equal(reduce(twice, { ...error, code: "error.internal" }).notices.length, 2);
  assert.equal(reduce(twice, { ...error, clip_id: "c2" }).notices.length, 2);
  assert.equal(reduce(twice, { type: "job.warning", code: "error.job.busy", params: {}, clip_id: null }).notices.length, 2);  // a warning is not that error
  const dismissed = reduce(twice, { type: "ui.notice.dismiss", id: twice.notices[0].id });
  assert.equal(reduce(dismissed, error).notices.length, 1);  // once dismissed it can come again
});

test("the clip that was selected comes back when it is still that file", () => {
  const base = reduce(initialState(), { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, job: null, clips: [summary("a", 1000), summary("b", 1000)] } });
  assert.equal(base.selectedId, "a");
  assert.equal(reduce(base, { type: "ui.restore", saved: { selected: { id: "b", name: "b.mp4" } } }).selectedId, "b");
  assert.equal(reduce(base, { type: "ui.restore", saved: { selected: { id: "b", name: "another-file.mp4" } } }).selectedId, "a");  // c-numbers are given again by a new run
  assert.equal(reduce(base, { type: "ui.restore", saved: { selected: { id: "zz", name: "zz.mp4" } } }).selectedId, "a");
});

test("a window reloaded in the middle of an export shows each clip's export as it is (v2.1.4)", () => {
  const analysis = { items: 1, to_check: 0, unchecked_frames: 0, rev: 1 };
  const done = { phase: "done", outputs: ["/o/a_anonymized.mp4"], report: { csv: "/o/a.csv" }, comparison: null, seconds: 3.2 };
  const state = run([{ type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, demo_limits: null,
    job: { job_id: "j2", kind: "render", state: "running" },
    clips: [summary("a", 100, { status: "rendered", analysis, render: done }),
            summary("b", 100, { status: "rendering", analysis, render: { phase: "render", fraction: 0.4 } }),
            summary("c", 100, { status: "queued", analysis, render: { phase: "queued", fraction: 0 } }),
            summary("d", 100, { status: "analyzed", analysis })] } }]);
  assert.deepEqual(state.clips.map((c) => c.status), ["analyzed", "rendering", "analyzed", "analyzed"]);  // a waiting clip is an analysed one, not a clip to analyse
  assert.deepEqual(state.clips.map((c) => c.render?.phase ?? null), ["done", "render", "queued", null]);
  assert.equal(state.clips[1].render.fraction, 0.4);
  assert.equal(state.clips[2].fraction, 1);
  assert.equal(overallFraction(state.clips, state.job) >= 0, true);
});

test("Analyze counts only the clips whose analysis was not made with these settings", () => {
  const made = { faces: true, plates: true, thoroughness: "standard", detection: "careful", gap_fill: "camera" };
  const analysis = (settings) => ({ items: 1, to_check: 0, unchecked_frames: 0, rev: 0, settings });
  const state = run([{ type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: {}, demo_limits: null, job: null, clips: [
    summary("done", 10, { status: "analyzed", analysis: analysis(made) }),
    summary("other", 10, { status: "analyzed", analysis: analysis({ ...made, detection: "fast" }) }),
    summary("changed", 10, { status: "analyzed", analysis: analysis(null) }),
    summary("new", 10),
    summary("busy", 10, { status: "analyzing" }),
  ] } }]);
  assert.deepEqual(clipsToAnalyze(state).map((c) => c.id), ["other", "changed", "new"]);
  const finished = reduce(state, { type: "clip.analyzed", clip_id: "new", items: 0, to_check: 0, unchecked_frames: 0, rev: 0, settings: made });
  assert.deepEqual(clipsToAnalyze(finished).map((c) => c.id), ["other", "changed"]);
  const other = reduce(finished, { type: "ui.options", change: { detection: "fast" } });
  assert.deepEqual(clipsToAnalyze(other).map((c) => c.id), ["done", "changed", "new"]);
});

test("many files refused for one reason are one notice and the first in full, a few stay as they are", () => {
  const heic = (n) => ({ path: `/x/IMG_${n}.HEIC`, code: "error.media.heic", params: { name: `IMG_${n}.HEIC`, extension: ".heic" } });
  const few = refusalNotices([heic(1), heic(2), heic(3)]);
  assert.deepEqual(few.map((n) => n.code), ["error.media.heic", "error.media.heic", "error.media.heic"]);
  const many = refusalNotices([...Array(300).keys()].map(heic).concat([{ path: "/x/a.raf", code: "error.media.raw", params: { name: "a.raf" } }]));
  assert.deepEqual(many.map((n) => n.code), ["error.media.heic", "warning.import.same_reason", "error.media.raw"]);
  assert.deepEqual(many[1].params, { count: 299, names: "IMG_1.HEIC, IMG_2.HEIC, IMG_3.HEIC" });
  assert.deepEqual(refusalNotices([]), []);
  const folder = refusalNotices([{ path: "", code: "warning.import.hidden", params: { count: 4 } }]);
  assert.deepEqual(folder, [{ code: "warning.import.hidden", params: { count: 4 } }]);
});
