import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { readFileSync } from "fs";
import { createI18n } from "../../src/anonymizer/ui/static/js/i18n.js";
import { percentText } from "../../src/anonymizer/ui/static/js/format.js";
import { languageFromBrowser } from "../../src/anonymizer/ui/static/js/language.js";
import { requestOptions, exampleName } from "../../src/anonymizer/ui/static/js/export-logic.js";
import { sortDropped } from "../../src/anonymizer/ui/static/js/state.js";
import { DEFAULT_EXPORT } from "../../src/anonymizer/ui/static/js/state.js";

const seven = ["en", "es", "fr", "de", "pt", "it", "zh-Hans"];

test("the page follows the browser's languages: the first the Studio has, a regional form counts as its language, English otherwise", () => {
  const follow = (...preferred) => languageFromBrowser(preferred, seven);
  assert.equal(follow("de-DE", "de"), "de");
  assert.equal(follow("fr"), "fr");
  assert.equal(follow("pt-BR"), "pt");
  assert.equal(follow("pt-PT", "en"), "pt");
  assert.equal(follow("es-MX"), "es");
  assert.equal(follow("it-CH"), "it");
  assert.equal(follow("zh-CN"), "zh-Hans");
  assert.equal(follow("zh-Hans-CN"), "zh-Hans");
  assert.equal(follow("zh"), "zh-Hans");
  assert.equal(follow("ja", "ko", "de-AT"), "de");   // the first one it has, not the first
  assert.equal(follow("en-US", "de"), "en");         // English first is English
  assert.equal(follow("DE_de"), "de");
  assert.equal(follow("ja"), "en");                  // none it has
  assert.equal(follow("ja", "ru"), "en");
  assert.equal(follow(), "en");
  assert.equal(languageFromBrowser(undefined, seven), "en");
  assert.equal(languageFromBrowser(["", "und", "*", null, 7, "xx-YY"], seven), "en");
  assert.equal(languageFromBrowser(["de"], ["en", "es"]), "en");  // a server with fewer languages
});

test("an export carries the language of the page, which the PDF is written in", () => {
  const capabilities = { pdf_report: { available: true }, comparison_video: { available: true } };
  assert.equal(requestOptions(DEFAULT_EXPORT, capabilities, { language: "de" }).language, "de");
  assert.equal("language" in requestOptions(DEFAULT_EXPORT, capabilities, { language: undefined }), false);
});

test("a dropped folder is sorted before anything is sent: media goes up, a known unsupported format is refused with its reason, the rest is counted", () => {
  const formats = { photo: [".jpg", ".png"], video: [".mp4"], unsupported: { ".heic": "error.media.heic", ".dng": "error.media.raw", ".bin": "error.media.format" }, output_folder: "anonymized" };
  const folder = (name) => ({ name, fromFolder: true });
  const sorted = sortDropped([folder("a.JPG"), folder("b.mp4"), folder("c.heic"), folder("d.zip"), folder("archive.7z"), folder("README"), folder("e.dng")], formats);
  assert.deepEqual(sorted.upload.map((e) => e.name), ["a.JPG", "b.mp4"]);
  assert.deepEqual(sorted.refused.map((r) => [r.code, r.path]), [["error.media.heic", "c.heic"], ["error.media.raw", "e.dng"], ["warning.import.ignored", ""]]);
  assert.deepEqual(sorted.refused[0].params, { name: "c.heic", extension: ".heic" });
  assert.deepEqual(sorted.refused[2].params, { count: 3, extensions: "(none), .7z, .zip" });
  // the same name in two subfolders is two files
  assert.equal(sortDropped([folder("notes.txt"), folder("notes.txt"), folder("a.jpg")], formats).refused[0].params.count, 2);
  // a file dropped by itself is sent: the server says why it cannot take it
  assert.deepEqual(sortDropped([{ name: "notes.txt", fromFolder: false }, { name: "c.heic", fromFolder: false }], formats).upload.length, 2);
  // an old server that tells no formats: everything goes up, as it did
  assert.equal(sortDropped([folder("d.zip")], null).upload.length, 1);
  assert.deepEqual(sortDropped([], formats), { upload: [], refused: [] });
});

test("a percentage is written the way the page's language writes it, on every screen that shows one (the second look score and a clip's progress had a fixed space)", () => {
  const language = (code) => createI18n(JSON.parse(readFileSync(new URL(`../../src/anonymizer/ui/i18n/${code}.json`, import.meta.url), "utf8")), code);
  assert.equal(percentText(language("en").t, 0.864), "86%");
  assert.equal(percentText(language("de").t, 0.864), "86 %");
  assert.equal(percentText(language("fr").t, 0.17), "17 %");
  assert.equal(percentText(language("zh-Hans").t, 1), "100%");
  assert.equal(percentText(language("es").t, 0), "0%");
});

test("the example of an output name has the extension in lower case, as the server writes it", () => {
  assert.equal(exampleName("{name}_anonymized", { name: "Westminster.JPG", media: "photo" }, 1), "Westminster_anonymized.jpg");
  assert.equal(exampleName("{name}_{index:03d}", { name: "a.Jpeg", media: "photo" }, 2), "a_002.jpeg");
  assert.equal(exampleName("{name}", { name: "Clip.MOV", media: "video" }, 1), "Clip.mp4");
});
