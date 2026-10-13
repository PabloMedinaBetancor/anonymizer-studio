import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { readFileSync } from "fs";
import { createI18n } from "../../src/anonymizer/ui/static/js/i18n.js";
import { initialState, reduce } from "../../src/anonymizer/ui/static/js/state.js";
import { errorNotice } from "../../src/anonymizer/ui/static/js/api.js";

const seven = ["en", "es", "fr", "de", "pt", "it", "zh-Hans"];
const texts = (code) => JSON.parse(readFileSync(new URL(`../../src/anonymizer/ui/i18n/${code}.json`, import.meta.url)));

test("a request that did not get an answer carries no reason: the browser's own English text is not shown", () => {
  const notice = errorNotice(new TypeError("NetworkError when attempting to fetch resource."));
  assert.equal(notice.code, "error.network");
  assert.deepEqual(notice.params, {});
  for (const code of seven) assert.ok(!/\{reason\}/.test(texts(code)["error.network"]), code);
});

test("the Studio gone and back is one flag of the state, and not a notice that goes by", () => {
  const lost = reduce(initialState(), { type: "server.lost", status: "gone" });
  assert.equal(lost.lost, "gone");
  assert.equal(lost.notices.length, 0);
  assert.equal(reduce(lost, { type: "server.found" }).lost, false);
  assert.equal(initialState().lost, false);
});

test("a page the Studio refuses (an address without its session) is 'forbidden': a bar, not a notice that goes after 20 s", () => {
  const refused = reduce(initialState(), { type: "job.error", code: "error.request.forbidden", params: {}, clip_id: null });
  assert.equal(refused.lost, "forbidden");
  assert.equal(refused.notices.length, 0);
  assert.equal(reduce(refused, { type: "server.lost", status: "gone" }).lost, "gone");
  for (const code of seven) assert.ok(texts(code)["lost.title_forbidden"] && texts(code)["lost.title"] && texts(code)["lost.how_demo"], code);
});

test("a stopped or failed clip does not say 'done' on the Analyze screen's rows", async () => {
  const keys = ["analyze.stopped", "analyze.failed", "analyze.queued", "analyze.not_analyzed", "analyze.some_done"];
  for (const code of seven) for (const key of keys) assert.ok(texts(code)[key], `${code} ${key}`);
  assert.ok(/\{done\}/.test(texts("en")["analyze.some_done"]) && /\{total\}/.test(texts("en")["analyze.some_done"]));
});

test("a clip written while the person stopped its comparison video keeps the word, and a finished one does not have it", () => {
  const base = reduce(initialState(), { type: "clips.added", clips: [{ clip_id: "c1", name: "a.mp4", media: "video", status: "analyzed", frames: 10 }] });
  const stopped = reduce(base, { type: "clip.rendered", clip_id: "c1", outputs: ["/o/a.mp4"], report: {}, comparison_stopped: true, seconds: 3 });
  assert.equal(stopped.clips[0].render.comparison_stopped, true);
  const whole = reduce(base, { type: "clip.rendered", clip_id: "c1", outputs: ["/o/a.mp4"], report: {}, comparison: "/o/a.compare.mp4", seconds: 3 });
  assert.equal(whole.clips[0].render.comparison_stopped, false);
});

test("'not possible in the current state of the file' says what the file is doing, in the seven languages", () => {
  for (const code of seven) {
    const { t } = createI18n(texts(code), code);
    const generic = t("error.clip.state", {});
    for (const status of ["analyzing", "queued", "rendering", "imported", "stopped", "failed", "idle"]) {
      const text = t("error.clip.state", { clip_id: "c1", status });
      assert.notEqual(text, generic, `${code} ${status}`);
      assert.ok(!text.startsWith("["), `${code} ${status}: ${text}`);
    }
    assert.equal(t("error.clip.state", { status: "somethingnew" }), generic);  // a state the page does not know falls back to the sentence without it
    assert.ok(!/state|estado|Zustand|\u00e9tat|stato/i.test(generic), `${code}: ${generic}`);
  }
});

test("the paper of the PDF report is A4, and Letter only for a browser in US English", async () => {
  const { paperFor, requestOptions } = await import("../../src/anonymizer/ui/static/js/export-logic.js");
  for (const locale of ["en-US", "en-us", "en-US-x-foo"]) assert.equal(paperFor(locale), "letter", locale);
  for (const locale of ["en", "en-GB", "en-CA", "es-ES", "de-DE", "pt-BR", "zh-Hans-CN", "", undefined, null]) assert.equal(paperFor(locale), "a4", String(locale));
  const exportOptions = { destination: "", name_pattern: "{name}", reports: { pdf: true }, comparison: false };
  assert.equal(requestOptions(exportOptions, {}, {}, "en-US").paper, "letter");
  assert.equal(requestOptions(exportOptions, {}, {}, "fr-FR").paper, "a4");
});

test("the note on the Analyze screen does not promise that a clip opens in Review by itself (nothing does: its card is clicked)", () => {
  assert.ok(!/opens in Review/.test(texts("en")["analyze.note"]));
  assert.ok(/click its card/.test(texts("en")["analyze.note"]));
});

test("a stopped export says how far it got; a stopped analysis and a finished export say nothing more", () => {
  const exporting = { ...initialState(), job: { id: "j1", kind: "render", state: "stopping" } };
  const stopped = reduce(exporting, { type: "job.finished", job_id: "j1", status: "stopped", summary: { clips: 3, rendered: 1 } });
  assert.equal(stopped.job, null);
  assert.equal(stopped.notices.length, 1);
  assert.equal(stopped.notices[0].code, "warning.export.stopped");
  assert.deepEqual(stopped.notices[0].params, { done: 1, total: 3 });
  for (const code of seven) assert.ok(/\{done\}/.test(texts(code)["warning.export.stopped"]) && /\{total\}/.test(texts(code)["warning.export.stopped"]), code);
  const analyzing = { ...initialState(), job: { id: "j2", kind: "analyze", state: "stopping" } };
  assert.equal(reduce(analyzing, { type: "job.finished", job_id: "j2", status: "stopped", summary: { clips: 3, analyzed: 1 } }).notices.length, 0);
  assert.equal(reduce(exporting, { type: "job.finished", job_id: "j1", status: "done", summary: { clips: 3, rendered: 3 } }).notices.length, 0);
});
