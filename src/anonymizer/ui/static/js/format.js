export function clock(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function timecode(seconds, fps) {
  const rate = Math.round(fps) || 1;
  const frames = Math.max(0, Math.round(seconds * fps));
  const whole = Math.floor(frames / rate);
  const h = String(Math.floor(whole / 3600)).padStart(2, "0");
  const m = String(Math.floor((whole % 3600) / 60)).padStart(2, "0");
  const s = String(whole % 60).padStart(2, "0");
  const f = String(frames % rate).padStart(2, "0");
  return `${h}:${m}:${s}:${f}`;
}

export function frameToSeconds(frame, fps) {
  return frame / fps;
}

// Duration estimates are shown as {key, params} so the sentence stays translatable.
export function durationParts(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return null;
  if (seconds < 60) return { key: "duration.under_minute", params: {} };
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return { key: "duration.minutes", params: { min: minutes } };
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (m === 0) return { key: "duration.hours", params: { h } };
  return { key: "duration.hours_minutes", params: { h, min: m } };
}

export function resolutionLabel(width, height) {
  const long = Math.max(width, height);
  const short = Math.min(width, height);
  if (long >= 3800) return "4K";
  return `${short}p`;
}

// The numbers are written as the page's language writes them: "2,8 MB" in German, French, Spanish, Italian and Portuguese, "2.8 MB" in English
// and Chinese. main.js says which language the page is in; until then (and in the tests) it is English.
let numberLocale = "en";
export function setNumberLocale(code) {
  try {
    new Intl.NumberFormat(code);
    numberLocale = code;
  } catch {
    numberLocale = "en";
  }
}

// A fraction as a percentage, written as the page's language writes it: "83%" in English, "83 %" in German and French (the text `review.cert.sure`,
// which the PDF report uses too). A figure with its own "%" in the code reads "83 %" in English and "83%" in German.
export function percentText(t, fraction) {
  return t("review.cert.sure", { percent: Math.round(fraction * 100) });
}

export function fixed(value, digits) {
  return new Intl.NumberFormat(numberLocale, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

export function whole(value) {
  return new Intl.NumberFormat(numberLocale).format(value);
}

// How long an export took: a tenth of a second under 10 s ("0.1 s", not "0 s" for a photo that took 0.1), whole seconds above.
export function secondsText(seconds) {
  const value = Math.max(Number.isFinite(seconds) ? seconds : 0, 0.1);
  const tenths = Math.round(value * 10) / 10;
  return tenths < 10 ? fixed(tenths, 1) : whole(Math.round(value));
}

export function megabytes(bytes) {
  if (bytes == null) return null;
  if (bytes >= 1e9) return `${fixed(bytes / 1e9, 1)} GB`;
  if (bytes >= 1e7) return `${whole(Math.round(bytes / 1e6))} MB`;
  if (bytes >= 1e6) return `${fixed(bytes / 1e6, 1)} MB`;
  return `${whole(Math.max(1, Math.round(bytes / 1e3)))} KB`;
}

// The clock on the live badge: a photo has no time, so it has none (its frame rate is 0, which made "NaN").
export function liveClock(clip) {
  if (clip.media !== "video" || !(clip.fps > 0)) return "";
  return timecode(clip.frame / clip.fps, clip.fps).slice(3, 8);
}

// "Frame 12 of 480" for a video; nothing for a photo, which is one frame.
export function frameLabel(clip, t) {
  if (clip.media !== "video") return "";
  return t("analyze.frame_of", { frame: whole(clip.frame), total: whole(clip.frames) });
}

// An estimated size as translatable parts: one figure when the ends are within a fifth of each other, else "low to high".
export function sizeParts(low, high) {
  if (low == null || high == null || high <= 0) return null;
  if (low >= high * 0.8) return { key: "size", params: { size: megabytes(high) } };
  return { key: "range", params: { low: megabytes(low), high: megabytes(high) } };
}
