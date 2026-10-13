import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { readFileSync, readdirSync, statSync } from "fs";
import { join } from "path";
import { performance } from "perf_hooks";
import { createI18n, pseudoText, pseudoDictionary } from "../../src/anonymizer/ui/static/js/i18n.js";

const uiRoot = new URL("../../src/anonymizer/ui/", import.meta.url).pathname;
const en = JSON.parse(readFileSync(join(uiRoot, "i18n/en.json"), "utf8"));

test("placeholders are filled and unknown ones left visible", () => {
  const { t } = createI18n({ a: "{x} of {y}" });
  assert.equal(t("a", { x: 1, y: 3 }), "1 of 3");
  assert.equal(t("a", { x: 1 }), "1 of {y}");
});

test("a missing key shows itself in brackets", () => {
  assert.equal(createI18n({}).t("nope"), "[nope]");
});

test("count selects the plural form", () => {
  const { t } = createI18n({ c: "{count} clips", "c.one": "{count} clip" });
  assert.equal(t("c", { count: 1 }), "1 clip");
  assert.equal(t("c", { count: 3 }), "3 clips");
});

function sources(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sources(path);
    return path.endsWith(".js") ? [path] : [];
  });
}

test("every literal key used by the pages exists in en.json", () => {
  const missing = [];
  for (const file of sources(join(uiRoot, "static/js"))) {
    const text = readFileSync(file, "utf8");
    for (const match of text.matchAll(/\bt\("([a-z_]+(?:\.[a-z_]+)+)"/g)) {
      if (!(match[1] in en) && !(`${match[1]}.one` in en)) missing.push(`${file}: ${match[1]}`);
    }
  }
  assert.deepEqual(missing, []);
});

const placeholders = (text) => [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
const es = JSON.parse(readFileSync(join(uiRoot, "i18n/es.json"), "utf8"));

test("Spanish has exactly the keys of English, with the same placeholders", () => {
  assert.deepEqual(Object.keys(es).sort(), Object.keys(en).sort());
  const wrong = Object.keys(en).filter((key) => placeholders(en[key]).join() !== placeholders(es[key]).join());
  assert.deepEqual(wrong, []);
});

test("Spanish is translated: only words that are the same in both languages stay equal", () => {
  const sameInBoth = new Set([
    "app.name", "analyze.clip", "analyze.row_clip", "export.clips_heading", "export.clips_heading.one", "export.report.json", "options.audio",
    "review.area.seconds", "review.cert.sure", "review.clips", "review.view.original", "review.visible",
  ]);
  const same = Object.keys(en).filter((key) => en[key] === es[key] && /[a-z]{4,}/i.test(en[key]) && !sameInBoth.has(key));
  assert.deepEqual(same, []);
});

test("the pseudo-language keeps placeholders, adds 40% and marks the ends", () => {
  const text = pseudoText("Render {count} clips");
  assert.deepEqual(placeholders(text), ["count"]);
  assert.ok(text.startsWith("[") && text.endsWith("]"));
  assert.ok(text.length >= "Render {count} clips".length * 1.4);
  assert.equal(pseudoText("{x}"), "[{x}]");
  const all = pseudoDictionary(en);
  assert.equal(Object.keys(all).length, Object.keys(en).length);
  const { t } = createI18n(all);
  assert.ok(t("analyze.cta", { count: 3 }).includes("3"));
});

test("the demo's limit is named in the page's language, not in the server's English word (v2.1.4)", () => {
  for (const lang of ["en", "es", "fr", "de", "it", "pt", "zh-Hans"]) {
    const dict = JSON.parse(readFileSync(join(uiRoot, `i18n/${lang}.json`), "utf8"));
    const { t } = createI18n(dict, lang);
    for (const limit of ["photos", "clips", "seconds", "height"]) {
      const said = t("error.demo.limit", { limit, max: 10 });
      assert.ok(said.includes(`(${dict[`demo.limit.${limit}`]},`) || said.includes(`（${dict[`demo.limit.${limit}`]}，`), `${lang} ${limit}: ${said}`);
      assert.ok(said.includes("10") && !said.includes("{"), said);
    }
  }
  assert.equal(createI18n({ "error.demo.limit": "over ({limit})" }).t("error.demo.limit", { limit: "other" }), "over (other)");  // a word nobody translated stays as it came
});
