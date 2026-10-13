// Prints, as JSON, the request the page's own code builds for each call that carries a body: api.js turns the call into a request and
// the screens' own builders (state.js analyzeOptions, export-logic.js renderRequest and planBody) make the options, the same functions
// views/import.js and views/export.js call. tests/test_page_server_contract.py sends every one of them to the real server and
// checks that views/export.js and views/import.js build their requests with those functions and nowhere else.
// Run: node tests/ui/capture_bodies.mjs
import { createHttpApi } from "../../src/anonymizer/ui/static/js/api.js";
import { DEFAULT_OPTIONS, analyzeOptions, initialState, reduce } from "../../src/anonymizer/ui/static/js/state.js";
import { planBody, renderRequest, renderable } from "../../src/anonymizer/ui/static/js/export-logic.js";

const sent = [];
globalThis.fetch = async (url, init) => {
  sent.push({ method: init.method, url, body: init.body === undefined ? null : JSON.parse(init.body) });
  return { ok: true, text: async () => "{}" };
};

const api = createHttpApi();
const capabilities = { pdf_report: { available: true }, comparison_video: { available: true } };
const clip = {
  clip_id: "CLIP", name: "f.png", media: "photo", source_dir: "/media", duration_s: 0, frames: 1, width: 320, height: 240, fps: 0,
  has_audio: false, has_gps: false, status: "analyzed", analysis: { items: 1, to_check: 1, unchecked_frames: 0, rev: 1 },
};
let state = reduce(initialState(), { type: "state.loaded", settings: {}, state: { mode: "local", languages: ["en"], capabilities, demo_limits: null, job: null, clips: [clip] } });
state = reduce(state, { type: "ui.options", change: { blur: "pixelate", strength: 1.5, extra_margin: 0.1, remove_gps: false, gap_fill: "hold" } });
state = reduce(state, { type: "export.options", change: { name_pattern: "{name}_out", destination: "DEST", comparison: false, reports: { pdf: false, csv: true, json: true } } });
const clips = renderable(state, null);
const ids = clips.map((c) => c.id);

await api.putSettings(state.options);
await api.analyze(ids, analyzeOptions(state.options));
await api.renderPlan(ids, planBody(state, ids).options);
const request = renderRequest(state, clips);
await api.render(request.ids, request.options);
await api.setItem("CLIP", 1, true);
await api.addArea("CLIP", { frame_from: 0, frame_to: 0, box: [10, 10, 40, 40], follow: "camera" }); // as views/review.js
await api.importPaths(["PATH"]);
console.log(JSON.stringify(sent));
