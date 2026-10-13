// Studio v2.1.1 (qa-anon, 04/10/2026): the notices that carry a count read right for one ("1 files were not recognized", "1 hidden entries",
// "left 1 places unchecked" were what a client with a single stray file or one skipped place was shown).
import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { readFileSync } from "fs";
import { join } from "path";
import { createI18n } from "../../src/anonymizer/ui/static/js/i18n.js";

const uiRoot = new URL("../../src/anonymizer/ui/", import.meta.url).pathname;
const LANGUAGES = ["en", "de", "es", "fr", "it", "pt", "zh-Hans"];
const dictionary = (lang) => JSON.parse(readFileSync(join(uiRoot, `i18n/${lang}.json`), "utf8"));
const COUNTED = ["warning.import.ignored", "warning.import.hidden", "warning.second_look.zones_skipped", "warning.vehicles.skipped"];

test("English says one file, one entry, one place for a count of one", () => {
  const { t } = createI18n(dictionary("en"), "en");
  assert.equal(t("warning.import.ignored", { count: 1, extensions: ".txt" }), "1 file was not recognized and was left out (.txt).");
  assert.match(t("warning.import.hidden", { count: 1 }), /^1 hidden entry \(/);
  assert.match(t("warning.second_look.zones_skipped", { count: 1 }), /left 1 place unchecked\. That place keeps/);
  assert.match(t("warning.second_look.zones_skipped", { count: 3 }), /left 3 places unchecked\. Those places keep/);
});

test("every language has a form for one of each counted notice, with no placeholder left unfilled", () => {
  for (const lang of LANGUAGES) {
    const dict = dictionary(lang);
    const { t } = createI18n(dict, lang);
    for (const key of COUNTED) {
      assert.ok(`${key}.one` in dict, `${lang}: ${key}.one is missing`);
      assert.ok(!t(key, { count: 1, extensions: ".txt" }).includes("{"), `${lang}: ${key} for 1`);
      assert.ok(t(key, { count: 4, extensions: ".txt" }).includes("4"), `${lang}: ${key} for 4`);
    }
  }
});
