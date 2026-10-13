import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { clock, timecode, durationParts, resolutionLabel, megabytes, setNumberLocale, fixed } from "../../src/anonymizer/ui/static/js/format.js";

test("clock drops the hour when it is zero", () => {
  assert.equal(clock(108), "01:48");
  assert.equal(clock(3725), "1:02:05");
});

test("timecode counts frames within the second", () => {
  assert.equal(timecode(31.4, 30), "00:00:31:12");
  assert.equal(timecode(0, 29.97), "00:00:00:00");
});

test("durations round to minutes and split hours", () => {
  assert.deepEqual(durationParts(20), { key: "duration.under_minute", params: {} });
  assert.deepEqual(durationParts(6 * 60 + 10), { key: "duration.minutes", params: { min: 6 } });
  assert.deepEqual(durationParts(70 * 60), { key: "duration.hours_minutes", params: { h: 1, min: 10 } });
  assert.deepEqual(durationParts(2 * 3600), { key: "duration.hours", params: { h: 2 } });
  assert.equal(durationParts(null), null);
});

test("resolution label uses the short side, 4K for UHD", () => {
  assert.equal(resolutionLabel(1920, 1080), "1080p");
  assert.equal(resolutionLabel(1080, 1920), "1080p");
  assert.equal(resolutionLabel(3840, 2160), "4K");
});

test("megabytes switches to GB above one billion bytes", () => {
  assert.equal(megabytes(212e6), "212 MB");
  assert.equal(megabytes(1.1e9), "1.1 GB");
  assert.equal(megabytes(null), null);
  assert.equal(megabytes(734691), "735 KB");
  assert.equal(megabytes(1.6e6), "1.6 MB");
});

test("sizes and counts are written as the page's language writes them", () => {
  setNumberLocale("de");
  assert.equal(megabytes(2.84e6), "2,8 MB");
  assert.equal(megabytes(1.5e9), "1,5 GB");
  assert.equal(fixed(23.97, 2), "23,97");
  setNumberLocale("zh-Hans");
  assert.equal(megabytes(2.84e6), "2.8 MB");
  setNumberLocale("not a locale");  // an unknown code leaves English
  assert.equal(megabytes(2.84e6), "2.8 MB");
  setNumberLocale("en");
});
