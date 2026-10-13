// Pure logic of the Export page: which clips can leave, what the request looks like, the file name example.
import { analyzedClips, uncheckedFrames } from "./state.js";

export const DEFAULT_PATTERN = "{name}_anonymized";

// A pattern is refused by the server when it holds a path separator, a character that Windows cannot save in a name, or nothing that names the file.
export function patternError(pattern) {
  if (!pattern || pattern.trim() === "") return "export.pattern.empty";
  if (/[\\/]/.test(pattern)) return "export.pattern.separator";
  if (/[<>:"|?*]/.test(pattern.replace(/\{[^}]*\}/g, ""))) return "export.pattern.characters";  // a name that Windows cannot save
  if (!/\{(name|index(:0?\d{0,2}d)?)\}/.test(pattern)) return "export.pattern.unique";
  return null;
}

export function splitName(fileName) {
  const dot = fileName.lastIndexOf(".");
  return dot > 0 ? { stem: fileName.slice(0, dot), ext: fileName.slice(dot) } : { stem: fileName, ext: "" };
}

// The example shown while typing; the real name comes back from the plan (it may get a -2 suffix).
export function exampleName(pattern, clip, index) {
  const { stem, ext } = splitName(clip.name);
  // {index} may carry a number format as the server reads it: {index:03d} is 001, {index:3d} is "  1".
  const base = pattern.split("{name}").join(stem).replace(/\{index(?::(0?)(\d{0,2})d)?\}/g, (_, zero, width) => String(index).padStart(Number(width || 0), zero ? "0" : " "));
  return `${base}${clip.media === "video" ? ".mp4" : ext.toLowerCase()}`;  // the server writes a photo's extension in lower case
}

// The number {index} stands for: the clip's place among the clips that will be written (the server numbers them so), 1 when it is not one of them.
export function positionAmong(ids, clip) {
  const place = ids.indexOf(clip.id);
  return place >= 0 ? place + 1 : 1;
}

// Clips shown in the list: every analyzed clip. The page never offers an unanalyzed one.
export function exportableClips(state) {
  return analyzedClips(state);
}

export function isIncluded(state, clip) {
  return !state.export.excluded.includes(clip.id);
}

export function blockedBy(clip, planEntry) {
  if (uncheckedFrames(clip) > 0) return { code: "error.analysis.incomplete", params: { frames: uncheckedFrames(clip) } };
  return planEntry?.blocked ?? null;
}

// The clips that would be rendered: included and not blocked.
export function renderable(state, plan) {
  const entries = new Map((plan?.clips ?? []).map((entry) => [entry.clip_id, entry]));
  return exportableClips(state).filter((clip) => isIncluded(state, clip) && blockedBy(clip, entries.get(clip.id)) == null);
}

export function toCheckTotal(clips) {
  return clips.reduce((sum, clip) => sum + (clip.analysis?.to_check ?? 0), 0);
}

// What goes to /api/render/plan and /api/render. An empty destination is left out so the server picks its default. The look of the
// blur (and the location choice, and the language of the page, which the PDF is written in) travel with every request: the page saves its settings a moment after they change, and an export
// started in that moment must still use what is on the screen.
const RENDER_SETTINGS = ["blur", "strength", "extra_margin", "remove_gps", "gap_fill", "language"];

// The paper of the PDF report: A4, except for a browser in US English (Letter there; Chromium prints Letter for everyone when the page does not say).
export function paperFor(locale) {
  return /^en-US\b/i.test(locale ?? "") ? "letter" : "a4";
}

export function requestOptions(exportOptions, capabilities, settings = {}, locale = globalThis.navigator?.language) {
  const reports = { ...exportOptions.reports };
  if (!capabilities?.pdf_report?.available) reports.pdf = false;
  const options = { name_pattern: exportOptions.name_pattern, reports, comparison: Boolean(exportOptions.comparison && capabilities?.comparison_video?.available) };
  if (exportOptions.destination.trim() !== "") options.destination = exportOptions.destination.trim();
  for (const key of RENDER_SETTINGS) if (settings[key] !== undefined) options[key] = settings[key];
  options.paper = paperFor(locale);
  return options;
}

// What the Export screen sends when the person starts a render of `clips` (the ones it will render): the ids and the options.
// views/export.js calls exactly this, so a test that builds the request builds the page's own.
export function renderRequest(state, clips) {
  return {
    ids: clips.map((clip) => clip.id),
    options: requestOptions(state.export.options, state.info?.capabilities, state.options),
  };
}

export function planBody(state, ids) {
  return { clips: ids, options: requestOptions(state.export.options, state.info?.capabilities, state.options) };
}

export function totalBytes(plan, ids) {
  const wanted = new Set(ids);
  return (plan?.clips ?? []).filter((entry) => wanted.has(entry.clip_id)).reduce((sum, entry) => sum + (entry.estimate_bytes ?? 0), 0);
}

// The sum of the low ends of the same estimates (the bytes go from this to totalBytes).
export function totalLowBytes(plan, ids) {
  const wanted = new Set(ids);
  return (plan?.clips ?? []).filter((entry) => wanted.has(entry.clip_id)).reduce((sum, entry) => sum + (entry.estimate_low_bytes ?? entry.estimate_bytes ?? 0), 0);
}

export function fitsOnDisk(plan, bytes) {
  return plan?.free_bytes == null || bytes <= plan.free_bytes;
}

// Overall fraction of a render job: each clip counts equally, a comparison video is a second pass for the clip.
export function renderFraction(clips, withComparison) {
  const active = clips.filter((clip) => clip.render);
  if (active.length === 0) return 0;
  const done = (clip) => {
    const r = clip.render;
    if (r.phase === "done") return 1;
    if (r.phase === "queued") return 0;
    if (!withComparison) return r.fraction;
    return r.phase === "render" ? r.fraction * 0.7 : 0.7 + r.fraction * 0.3;
  };
  return active.reduce((sum, clip) => sum + done(clip), 0) / active.length;
}

// Files of a finished clip, as [kind, path] in a stable order.
export function resultFiles(render) {
  if (!render || render.phase !== "done") return [];
  const files = (render.outputs ?? []).map((path) => ["output", path]);
  for (const kind of ["pdf", "csv", "json"]) if (render.report?.[kind]) files.push([kind, render.report[kind]]);
  if (render.comparison) files.push(["comparison", render.comparison]);
  return files;
}

export function fileLabel(path) {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}
