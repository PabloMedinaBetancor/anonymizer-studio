import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { initialState, reduce } from "../../src/anonymizer/ui/static/js/state.js";
import { patternError, exampleName, renderable, requestOptions, totalBytes, fitsOnDisk, renderFraction, resultFiles, fileLabel, blockedBy, renderRequest, positionAmong } from "../../src/anonymizer/ui/static/js/export-logic.js";

const analysis = (extra = {}) => ({ items: 3, to_check: 0, unchecked_frames: 0, rev: 1, ...extra });
const summary = (id, extra = {}) => ({ clip_id: id, name: `${id}.mov`, media: "video", source_dir: "/media/clips", duration_s: 10, frames: 250, width: 1920, height: 1080, fps: 25, has_audio: true, has_gps: false, status: "analyzed", analysis: analysis(), ...extra });
const load = (...clips) => reduce(initialState(), { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities: { pdf_report: { available: true }, comparison_video: { available: true } }, demo_limits: null, job: null, clips } });

test("patterns: a separator or a missing placeholder is refused", () => {
  assert.equal(patternError("{name}_anonymized"), null);
  assert.equal(patternError("clip_{index}"), null);
  assert.equal(patternError(""), "export.pattern.empty");
  assert.equal(patternError("a/{name}"), "export.pattern.separator");
  assert.equal(patternError("{name}: final"), "export.pattern.characters");
  assert.equal(patternError("clip_{index:03d}"), null);  // a format inside a field may hold a colon
  assert.equal(patternError("."), "export.pattern.unique");
  assert.equal(patternError("a\\{name}"), "export.pattern.separator");
  assert.equal(patternError("fixed"), "export.pattern.unique");
});

test("the example name follows a number format of {index}, as the server numbers", () => {
  const clip = { name: "a.jpg", media: "photo" };
  assert.equal(exampleName("p_{index:03d}", clip, 7), "p_007.jpg");
  assert.equal(exampleName("p_{index:d}", clip, 7), "p_7.jpg");
  assert.equal(exampleName("{name}-{index}", clip, 12), "a-12.jpg");
});

test("the example name takes the source stem and the output extension", () => {
  assert.equal(exampleName("{name}_anonymized", { name: "street.mov", media: "video" }, 1), "street_anonymized.mp4");
  assert.equal(exampleName("{name}_anonymized", { name: "plate.jpg", media: "photo" }, 1), "plate_anonymized.jpg");
  assert.equal(exampleName("{index}-{name}", { name: "a.b.mp4", media: "video" }, 3), "3-a.b.mp4");
});

test("a clip with unchecked frames is blocked and not renderable", () => {
  const state = load(summary("a"), summary("b", { analysis: analysis({ unchecked_frames: 12 }) }));
  assert.deepEqual(blockedBy(state.clips[1], null), { code: "error.analysis.incomplete", params: { frames: 12 } });
  assert.deepEqual(renderable(state, null).map((c) => c.id), ["a"]);
});

test("unticked clips and clips the plan blocks are left out; unanalyzed ones never listed", () => {
  let state = load(summary("a"), summary("b"), summary("c"), summary("d", { status: "imported", analysis: null }));
  state = reduce(state, { type: "export.include", clip_id: "b", value: false });
  const plan = { clips: [{ clip_id: "a", blocked: null }, { clip_id: "c", blocked: { code: "error.source.changed", params: {} } }] };
  assert.deepEqual(renderable(state, plan).map((c) => c.id), ["a"]);
  state = reduce(state, { type: "export.include", clip_id: "b", value: true });
  assert.deepEqual(renderable(state, null).map((c) => c.id), ["a", "b", "c"]);
});

test("request options drop the PDF and the comparison the server cannot make, and an empty destination", () => {
  const options = { destination: " ", name_pattern: "{name}_x", reports: { pdf: true, csv: true, json: false }, comparison: true };
  const without = requestOptions(options, { pdf_report: { available: false }, comparison_video: { available: false } }, {}, "de-DE");
  assert.deepEqual(without, { name_pattern: "{name}_x", reports: { pdf: false, csv: true, json: false }, comparison: false, paper: "a4" });
  const withAll = requestOptions({ ...options, destination: "/out" }, { pdf_report: { available: true }, comparison_video: { available: true } });
  assert.equal(withAll.destination, "/out");
  assert.equal(withAll.comparison, true);
  assert.equal(withAll.reports.pdf, true);
});

test("the look of the blur and the location choice go with the request, so an export started before the page saved them uses them", () => {
  const options = { destination: "", name_pattern: "{name}_x", reports: { pdf: false, csv: true, json: false }, comparison: false };
  const settings = { blur: "solid", strength: 1.5, extra_margin: 0.1, remove_gps: false, gap_fill: "hold", language: "es", faces: true };
  assert.deepEqual(requestOptions(options, {}, settings, "de-DE"), {
    name_pattern: "{name}_x", reports: { pdf: false, csv: true, json: false }, comparison: false, paper: "a4",
    blur: "solid", strength: 1.5, extra_margin: 0.1, remove_gps: false, gap_fill: "hold", language: "es",
  });
  assert.equal(requestOptions(options, {}).blur, undefined);
});

test("size totals count only the wanted clips and are checked against free space", () => {
  const plan = { free_bytes: 50, clips: [{ clip_id: "a", estimate_bytes: 30 }, { clip_id: "b", estimate_bytes: 40 }] };
  assert.equal(totalBytes(plan, ["a"]), 30);
  assert.equal(fitsOnDisk(plan, totalBytes(plan, ["a", "b"])), false);
  assert.equal(fitsOnDisk(plan, 30), true);
  assert.equal(fitsOnDisk({ free_bytes: null }, 1e12), true);
});

test("rendering: progress, results and the return to an analyzed clip", () => {
  let state = load(summary("a"), summary("b"));
  state = reduce(state, { type: "job.started", job_id: "j2", kind: "render", clips: [{ clip_id: "a" }, { clip_id: "b" }] });
  assert.equal(state.tab, "import");
  assert.equal(state.clips[0].render.phase, "queued");
  state = reduce(state, { type: "clip.started", clip_id: "a", phase: "render" });
  state = reduce(state, { type: "clip.progress", clip_id: "a", phase: "render", fraction: 0.5, frame: 125, frames: 250, eta_s: 3 });
  assert.equal(state.clips[0].status, "rendering");
  assert.equal(renderFraction(state.clips, false), 0.25);
  state = reduce(state, { type: "clip.progress", clip_id: "a", phase: "compare", fraction: 0.5, frame: 1, frames: 2, eta_s: null });
  assert.equal(state.clips[0].render.phase, "compare");
  state = reduce(state, { type: "clip.rendered", clip_id: "a", outputs: ["/out/a_anonymized.mp4"], report: { csv: "/out/a_anonymized.report.csv" }, comparison: "/out/a_anonymized.compare.mp4", seconds: 4 });
  assert.equal(state.clips[0].status, "analyzed");
  assert.deepEqual(resultFiles(state.clips[0].render).map(([k]) => k), ["output", "csv", "comparison"]);
  state = reduce(state, { type: "job.finished", job_id: "j2", status: "stopped", summary: {} });
  assert.equal(state.clips[0].render.phase, "done");
  assert.equal(state.clips[1].render, null);
  assert.equal(state.clips[1].status, "analyzed");
  assert.equal(state.job, null);
});

test("a rendered clip from the server is an analyzed clip for the page", () => {
  const state = load(summary("a", { status: "rendered" }));
  assert.equal(state.clips[0].status, "analyzed");
});

test("file labels are the last path part, also for URLs and Windows paths", () => {
  assert.equal(fileLabel("/api/downloads/f1/street_anonymized.mp4"), "street_anonymized.mp4");
  assert.equal(fileLabel(["Users", "a", "x.pdf"].join(String.fromCharCode(92))), "x.pdf");
});

test("the render request of the Export screen carries the ids and the options, nothing to acknowledge", () => {
  const state = load(summary("a", { analysis: analysis({ to_check: 2 }) }), summary("b"));
  const request = renderRequest(state, renderable(state, null));
  assert.deepEqual(request.ids, ["a", "b"]);
  assert.equal("extra" in request, false);
  assert.equal("acknowledge_visible" in request.options, false);
});

test("the number {index} stands for is the clip's place among the clips that will be written, as the server numbers them", () => {
  assert.equal(positionAmong(["b", "c"], { id: "b" }), 1);
  assert.equal(positionAmong(["b", "c"], { id: "c" }), 2);
  assert.equal(positionAmong(["b", "c"], { id: "a" }), 1);
  assert.equal(exampleName("p_{index}", { name: "b.jpg", media: "photo" }, positionAmong(["b", "c"], { id: "b" })), "p_1.jpg");
});
