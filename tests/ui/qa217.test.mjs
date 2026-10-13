import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { readFileSync } from "fs";
import { createI18n } from "../../src/anonymizer/ui/static/js/i18n.js";
import { certaintyKey, isPositionBand } from "../../src/anonymizer/ui/static/js/review-logic.js";
import { secondsText, setNumberLocale } from "../../src/anonymizer/ui/static/js/format.js";

const seven = ["en", "es", "fr", "de", "pt", "it", "zh-Hans"];
const band = (extra = {}) => ({ item_id: 1, kind: "plate", judgement: { verdict: "blur", reason: "plate.reason.position_band" }, max_score: 1, reviewed: false, manual: false, ...extra });

test("a plate band has no certainty: a dash, however it stands (guess, to check, decided by the person)", () => {
  assert.equal(isPositionBand(band()), true);
  assert.equal(isPositionBand({ ...band(), judgement: { verdict: "blur", reason: "plate.reason.vehicle_text" } }), false);
  assert.equal(certaintyKey(band()), "review.cert.band");
  assert.equal(certaintyKey(band({ judgement: { verdict: "check", reason: "plate.reason.position_band" } })), "review.cert.band_check");
  assert.equal(certaintyKey(band({ reviewed: true })), "review.cert.band_decided");
  assert.equal(certaintyKey(band({ manual: true })), "review.cert.manual");  // an area the person drew is theirs, whatever its reason
});

test("the texts of a plate band exist in the seven languages and none shows a percentage", () => {
  for (const code of seven) {
    const texts = JSON.parse(readFileSync(new URL(`../../src/anonymizer/ui/i18n/${code}.json`, import.meta.url)));
    const { t } = createI18n(texts);
    for (const key of ["review.cert.band", "review.cert.band_decided", "review.cert.band_check"]) {
      const text = t(key, { percent: 100 });
      assert.ok(text.startsWith("\u2013"), `${code} ${key}: ${text}`);
      assert.ok(!/100/.test(text), `${code} ${key}: ${text}`);
    }
    assert.ok(/1\.0/.test(t("report.note.band")), code);  // the footnote of the PDF says the CSV and JSON carry a constant 1.0
  }
});

test("an export that took a fraction of a second says so, in the page's own way of writing numbers", () => {
  setNumberLocale("en");
  assert.equal(secondsText(0.04), "0.1");
  assert.equal(secondsText(0.11), "0.1");
  assert.equal(secondsText(3.49), "3.5");
  assert.equal(secondsText(9.96), "10");
  assert.equal(secondsText(12.4), "12");
  assert.equal(secondsText(null), "0.1");
  setNumberLocale("de");
  assert.equal(secondsText(3.49), "3,5");
  setNumberLocale("en");
});
