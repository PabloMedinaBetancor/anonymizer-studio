// The options that /api/job/analyze reads: what is looked for and how hard. The look of the blur is not part of an analysis.
export function analyzeOptions(options) {
  const { faces, plates, thoroughness, detection, gap_fill } = options;
  return { faces, plates, thoroughness, detection, gap_fill };
}

// What decides how long a frame takes, as the server keys its measurements.
export function rateKey(options) {
  return `${options.thoroughness}/${options.detection}`;
}

export const DEFAULT_OPTIONS = {
  faces: true,
  plates: true,
  thoroughness: "standard",
  detection: "careful",
  gap_fill: "camera",
  blur: "gaussian",
  strength: 1.0,
  extra_margin: 0.0,
  remove_gps: true,
};

export const DEFAULT_EXPORT = {
  destination: "",
  name_pattern: "{name}_anonymized",
  reports: { pdf: true, csv: true, json: true },
  comparison: false,
};

export function initialState() {
  return {
    tab: "import",
    info: null,
    clips: [],
    selectedId: null,
    options: { ...DEFAULT_OPTIONS },
    secondsPerFrame: {},
    job: null,
    lost: false,  // "gone": the Studio does not answer (it stopped); "forbidden": it refuses this page (an address without its session): the page says so and offers nothing
    notices: [],
    review: { view: "anonymized", onlyCheck: false, onlySecond: false, itemId: null },
    export: { excluded: [], options: { ...DEFAULT_EXPORT } },
  };
}

function newClip(summary) {
  // An analysed clip that waits for its export is "queued" at the server (a window reloaded in the middle of an export meets it so).
  const waiting = summary.status === "queued" && summary.analysis != null && summary.render != null;
  const analyzed = summary.status === "analyzed" || summary.status === "rendering" || summary.status === "rendered" || waiting;
  // The server calls a rendered clip "rendered" until it is touched again; for the page it is still an analyzed clip.
  const status = summary.status === "rendered" || waiting ? "analyzed" : (summary.status ?? "imported");
  return {
    id: summary.clip_id,
    name: summary.name,
    media: summary.media,
    source_dir: summary.source_dir ?? null,
    duration_s: summary.duration_s,
    frames: summary.frames,
    width: summary.width,
    height: summary.height,
    fps: summary.fps,
    codec: summary.codec ?? null,
    has_audio: summary.has_audio,
    has_gps: summary.has_gps,
    size_bytes: summary.size_bytes ?? null,
    source_url: summary.source_url ?? null,
    thumb_url: summary.thumb_url ?? null,
    status,
    render: summary.render ?? null,  // how far its export is, or what it wrote (a reloaded window shows it as the other one does)
    analysis: summary.analysis ?? null,
    // A clip met in the middle of its analysis (this window was opened or reloaded late) comes with how far it is, what it has found
    // and, when it failed, why: the page does not show it as a clip at 0 % or one that failed for no reason.
    fraction: analyzed ? 1 : (summary.progress?.fraction ?? 0),
    frame: summary.progress?.frame ?? 0,
    eta_s: summary.progress?.eta_s ?? null,
    live: summary.progress?.live ?? [],
    found: summary.progress?.found ?? null,
    stage: summary.progress?.stage ?? null,
    zones: summary.progress?.zones ?? null,
    failure: summary.failure ?? null,
    items: summary.items_so_far ?? [],
  };
}

function withClip(state, clipId, change) {
  let touched = false;
  const clips = state.clips.map((clip) => {
    if (clip.id !== clipId) return clip;
    touched = true;
    return { ...clip, ...change(clip) };
  });
  return touched ? { ...state, clips } : state;
}

let noticeCounter = 0;

function addNotice(state, level, message) {
  // The same notice that is already on the page is not stacked again (a server that refuses every request says the same thing seven times).
  const params = JSON.stringify(message.params ?? {});
  const clip = message.clip_id ?? null;
  if (state.notices.some((n) => n.level === level && n.code === message.code && n.clip_id === clip && JSON.stringify(n.params) === params)) return state;
  noticeCounter += 1;
  const notice = { id: noticeCounter, level, code: message.code, params: message.params ?? {}, clip_id: message.clip_id ?? null };
  return { ...state, notices: [...state.notices, notice] };
}

// A decision this window already has (it made it, and the event came after the answer, or the other way round) is not applied twice.
function isOlder(state, message) {
  const analysis = state.clips.find((c) => c.id === message.clip_id)?.analysis;
  return analysis != null && analysis.rev > message.rev;
}

function notLoaded(state, clipId) {
  const clip = state.clips.find((c) => c.id === clipId);
  return clip != null && clip.items.length === 0 && (clip.analysis?.items ?? 0) > 0;
}

const WORKING = new Set(["queued", "analyzing", "rendering"]);

export function reduce(state, message) {
  switch (message.type) {
    case "ui.tab":
      return state.tab === message.tab ? state : { ...state, tab: message.tab };
    case "ui.select":
      return { ...state, selectedId: message.clip_id, review: { ...state.review, itemId: null } };
    case "review.open":
      return { ...state, tab: "review", selectedId: message.clip_id, review: { ...state.review, itemId: null } };
    case "review.view":
      return { ...state, review: { ...state.review, view: message.view } };
    case "review.only_check":
      return { ...state, review: { ...state.review, onlyCheck: message.value } };
    case "review.only_second":
      return { ...state, review: { ...state.review, onlySecond: message.value } };
    case "review.select":
      return state.review.itemId === message.item_id ? state : { ...state, review: { ...state.review, itemId: message.item_id } };
    case "item.changed":
      if (isOlder(state, message)) return state;
      return withClip(state, message.clip_id, (clip) => ({
        items: clip.items.map((item) => (item.item_id === message.item.item_id ? message.item : item)),
        analysis: clip.analysis && { ...clip.analysis, rev: message.rev, to_check: message.to_check },
      }));
    case "items.changed": {
      if (isOlder(state, message)) return state;
      const changed = new Map(message.items.map((item) => [item.item_id, item]));
      return withClip(state, message.clip_id, (clip) => ({
        items: clip.items.map((item) => changed.get(item.item_id) ?? item),
        analysis: clip.analysis && { ...clip.analysis, rev: message.rev, to_check: message.to_check },
      }));
    }
    case "area.added":
      // The same area comes twice when this window drew it: in the answer of its request and in the event every window gets.
      // A clip whose items this window has not loaded yet gets the area with the rest, from the server.
      if (notLoaded(state, message.clip_id) || state.clips.find((c) => c.id === message.clip_id)?.items.some((item) => item.item_id === message.item.item_id)) return state;
      return withClip(state, message.clip_id, (clip) => ({
        items: [...clip.items, message.item].sort((a, b) => a.first_frame - b.first_frame || a.item_id - b.item_id),
        analysis: clip.analysis && { ...clip.analysis, rev: message.rev, items: clip.analysis.items + 1 },
      }));
    case "area.removed": {
      if (!state.clips.find((c) => c.id === message.clip_id)?.items.some((item) => item.item_id === message.item_id)) return state;
      const next = withClip(state, message.clip_id, (clip) => ({
        items: clip.items.filter((item) => item.item_id !== message.item_id),
        analysis: clip.analysis && { ...clip.analysis, rev: message.rev, to_check: message.to_check, items: Math.max(0, clip.analysis.items - 1) },
      }));
      return state.review.itemId === message.item_id ? { ...next, review: { ...next.review, itemId: null } } : next;
    }
    case "ui.restore": {
      // What this window had on the Export screen before it was reloaded (or its language changed): each value is checked, the saved
      // text is the person's browser's and not a promise.
      const saved = message.saved?.export ?? {};
      const options = { ...state.export.options };
      if (typeof saved.options?.destination === "string") options.destination = saved.options.destination;
      if (typeof saved.options?.name_pattern === "string") options.name_pattern = saved.options.name_pattern;
      if (typeof saved.options?.comparison === "boolean") options.comparison = saved.options.comparison;
      for (const format of ["pdf", "csv", "json"]) {
        if (typeof saved.options?.reports?.[format] === "boolean") options.reports = { ...options.reports, [format]: saved.options.reports[format] };
      }
      const names = new Map(state.clips.map((clip) => [clip.id, clip.name]));
      const excluded = Array.isArray(saved.excluded)
        ? saved.excluded.filter((entry) => entry && names.get(entry.id) === entry.name).map((entry) => entry.id)
        : state.export.excluded;
      // The clip the person was looking at, when it is still that file.
      const selected = message.saved?.selected;
      const selectedId = selected && names.get(selected.id) === selected.name ? selected.id : state.selectedId;
      return { ...state, selectedId, export: { options, excluded } };
    }
    case "export.options":
      return { ...state, export: { ...state.export, options: { ...state.export.options, ...message.change } } };
    case "export.report":
      return { ...state, export: { ...state.export, options: { ...state.export.options, reports: { ...state.export.options.reports, [message.format]: message.value } } } };
    case "export.include": {
      const rest = state.export.excluded.filter((id) => id !== message.clip_id);
      return { ...state, export: { ...state.export, excluded: message.value ? rest : [...rest, message.clip_id] } };
    }
    case "ui.options":
      return { ...state, options: { ...state.options, ...message.change } };
    case "ui.notice.dismiss":
      return { ...state, notices: state.notices.filter((n) => n.id !== message.id) };
    case "estimate":
      return { ...state, secondsPerFrame: { ...state.secondsPerFrame, [message.key]: message.seconds_per_frame } };
    case "state.loaded": {
      const s = message.state;
      const clips = s.clips.map(newClip);
      const keep = clips.some((c) => c.id === state.selectedId);
      return {
        ...state,
        info: { formats: s.formats ?? null, idle: s.idle_stop ? { minutes: Math.round(s.idle_stop.seconds / 60), hours: Math.round(s.idle_stop.seconds_with_clips / 3600) } : null, mode: s.mode, languages: s.languages, capabilities: s.capabilities, demo_limits: s.demo_limits, costs: { careful: s.slowdown?.careful ?? null } },
        options: { ...state.options, ...message.settings },
        clips,
        selectedId: keep ? state.selectedId : (clips[0]?.id ?? null),
        job: s.job && { id: s.job.job_id, kind: s.job.kind, state: s.job.state },
      };
    }
    case "clips.added": {
      const known = new Set(state.clips.map((clip) => clip.id));
      const added = message.clips.filter((c) => !known.has(c.clip_id)).map(newClip);
      if (added.length === 0) return state;
      return { ...state, clips: [...state.clips, ...added], selectedId: state.selectedId ?? added[0].id };
    }
    case "clip.removed": {
      const clips = state.clips.filter((clip) => clip.id !== message.clip_id);
      const selectedId = clips.some((clip) => clip.id === state.selectedId) ? state.selectedId : (clips[0]?.id ?? null);
      return { ...state, clips, selectedId };
    }
    case "job.started": {
      const listed = new Set(message.clips.map((c) => c.clip_id));
      const analyzing = message.kind === "analyze";
      return {
        ...state,
        tab: analyzing ? "analyze" : state.tab,
        job: { id: message.job_id, kind: message.kind, state: "running" },
        clips: analyzing
          ? state.clips.map((c) => (listed.has(c.id) ? { ...c, status: "queued", fraction: 0, frame: 0, eta_s: null, live: [], stage: null, zones: null, found: null, items: [], analysis: null } : c))
          : state.clips.map((c) => (listed.has(c.id) ? { ...c, render: { phase: "queued", fraction: 0 } } : c)),
      };
    }
    case "clip.started":
      return withClip(state, message.clip_id, (clip) =>
        message.phase === "analyze" ? { status: "analyzing" } : { status: "rendering", render: { ...keepOutput(clip), phase: "render", fraction: 0 } });
    case "clip.progress":
      if (message.phase === "render" || message.phase === "compare") {
        return withClip(state, message.clip_id, (clip) => ({ status: "rendering", render: { ...keepOutput(clip), phase: message.phase, fraction: message.fraction } }));
      }
      if (message.phase !== "analyze") return state;
      return withClip(state, message.clip_id, (clip) => ({
        status: "analyzing",
        fraction: Math.max(clip.fraction, message.fraction),
        frame: message.frame,
        eta_s: message.eta_s,
        live: message.live ?? [],
        found: message.found ?? clip.found ?? null,
        stage: message.stage ?? "first",
        zones: message.zones ?? null,
      }));
    case "clip.item":
      return withClip(state, message.clip_id, (clip) =>
        clip.items.some((item) => item.item_id === message.item.item_id) ? {} : { items: [...clip.items, message.item] },
      );
    case "clip.analyzed":
      return withClip(state, message.clip_id, (clip) => ({
        status: "analyzed",
        frames: message.frames ?? clip.frames,  // the real count: the file's header can say one more than the decoder gives
        fraction: 1,
        eta_s: 0,
        live: [],
        stage: null,
        zones: null,
        analysis: { items: message.items, to_check: message.to_check, unchecked_frames: message.unchecked_frames, rev: message.rev, ...(message.settings ? { settings: message.settings } : {}) },
      }));
    case "clip.rendered":
      return withClip(state, message.clip_id, () => ({
        status: "analyzed",
        render: {
          phase: "done", outputs: message.outputs ?? [], report: message.report ?? {}, comparison: message.comparison ?? null,
          comparison_stopped: message.comparison_stopped === true, seconds: message.seconds ?? null,
        },
      }));
    case "clip.items.loaded":
      return withClip(state, message.clip_id, () => ({ items: message.items }));
    case "server.lost":
      return state.lost === message.status ? state : { ...state, lost: message.status ?? "gone" };
    case "server.found":
      return state.lost ? { ...state, lost: false } : state;
    case "job.warning":
      return addNotice(state, "warning", message);
    case "job.error": {
      // A page the Studio does not let in cannot do anything: the bar over it says so (and how), instead of a notice that goes after 20 s.
      if (message.code === "error.request.forbidden") return state.lost ? state : { ...state, lost: "forbidden" };
      const noticed = addNotice(state, "error", message);
      if (!message.clip_id) return noticed;
      // The clip that was being analyzed when the job reported its error did not stop on the person's word: it failed, and says why.
      return withClip(noticed, message.clip_id, (clip) =>
        clip.status === "analyzing" || clip.status === "queued"
          ? { status: "failed", failure: { code: message.code, params: message.params ?? {} }, live: [], stage: null, zones: null }
          : {});
    }
    case "job.state":
      return state.job ? { ...state, job: { ...state.job, state: message.state } } : state;
    case "job.finished": {
      // A stop of an export said nothing (the progress bar went and the clips not reached went back to their plan): the notice says how far it got.
      const said = state.job?.kind === "render" && message.status === "stopped"
        ? addNotice(state, "warning", { code: "warning.export.stopped", params: { done: message.summary?.rendered ?? 0, total: message.summary?.clips ?? 0 } })
        : state;
      return {
        ...said,
        job: null,
        clips: state.clips.map((clip) => {
          if (clip.status === "rendering" || (clip.render && clip.render.phase !== "done")) {
            return { ...clip, status: clip.status === "rendering" ? "analyzed" : clip.status, render: clip.render?.phase === "done" ? clip.render : null };
          }
          return WORKING.has(clip.status) ? { ...clip, status: message.status === "failed" ? "failed" : "stopped", live: [] } : clip;
        }),
      };
    }
    default:
      return state;
  }
}

// The file a running export is writing, when the server said (a window opened in the middle of the export is told; the events carry no name).
function keepOutput(clip) {
  return clip.render?.output ? { output: clip.render.output } : {};
}

export function isWorking(clip) {
  return WORKING.has(clip.status);
}

// What the server refused of an import, as notices: many files refused for one reason (a folder of 300 HEIC photos) are the first one in
// full and one line for the rest, not 300 lines that go by in a few seconds.
export function refusalNotices(rejected, limit = 3) {
  const byCode = new Map();
  for (const refused of rejected) {
    if (!byCode.has(refused.code)) byCode.set(refused.code, []);
    byCode.get(refused.code).push(refused);
  }
  const notices = [];
  for (const [code, list] of byCode) {
    for (const refused of list.length > limit ? list.slice(0, 1) : list) notices.push({ code, params: refused.params });
    if (list.length > limit) {
      const rest = list.slice(1);
      const names = rest.slice(0, 3).map((refused) => refused.params?.name ?? refused.params?.path ?? refused.path);
      notices.push({ code: "warning.import.same_reason", params: { count: rest.length, names: names.join(", ") } });
    }
  }
  return notices;
}

// What a dropped folder holds, sorted before anything is sent (it used to upload every file in it, a 3 GB archive included, for the server to
// refuse it): photos and videos go up; a format known not to be supported is refused here with the server's own reason (the same notice as the
// "+ Import" dialog gives, which sends paths and so copies nothing); any other file is left out and counted, with its extension.
// `entries` are {name, fromFolder, ...}: a file dropped by itself is always sent (the server says why it cannot take it); `formats` is the
// server's own list (`state.formats`); without it everything goes up, as it did.
export function sortDropped(entries, formats) {
  const upload = [];
  const refused = [];
  const kinds = new Set();
  let ignored = 0;
  for (const entry of entries) {
    const extension = (/\.[^./\\]*$/.exec(entry.name)?.[0] ?? "").toLowerCase();
    const reason = formats?.unsupported?.[extension];
    if (!formats || !entry.fromFolder || formats.photo.includes(extension) || formats.video.includes(extension)) upload.push(entry);
    else if (reason) refused.push({ path: entry.name, code: reason, params: { name: entry.name, extension } });
    else {
      ignored += 1;  // counted file by file: two `notes.txt` in two subfolders are two files
      kinds.add(extension || "(none)");
    }
  }
  if (ignored > 0) refused.push({ path: "", code: "warning.import.ignored", params: { count: ignored, extensions: [...kinds].sort().slice(0, 8).join(", ") } });
  return { upload, refused };
}

export function analyzableClips(state) {
  return state.clips.filter((clip) => !isWorking(clip));
}

// What "Analyze" would do now: the clips that are not being worked on and whose analysis, if any, was not made with these settings
// (the server skips a current analysis at once, so counting those misled: "Analyze 8 clips" for eight that were done).
export function clipsToAnalyze(state) {
  const wanted = analyzeOptions(state.options);
  return analyzableClips(state).filter((clip) => {
    const made = clip.analysis?.settings;
    return !made || Object.keys(wanted).some((field) => made[field] !== wanted[field]);
  });
}

export function selectedClip(state) {
  return state.clips.find((clip) => clip.id === state.selectedId) ?? null;
}

export function itemCounts(items) {
  const counts = { faces: 0, plates: 0, not_sure: 0 };
  for (const item of items) {
    if (item.judgement.verdict === "check") counts.not_sure += 1;
    else if (item.kind === "face") counts.faces += 1;
    else counts.plates += 1;
  }
  return counts;
}

// The estimate is whatever the server measured on earlier jobs of this session; before that there is no number.
export function estimateSeconds(clips, state) {
  const perFrame = state.secondsPerFrame[rateKey(state.options)];
  if (perFrame == null) return null;
  return clips.reduce((sum, clip) => sum + clip.frames * perFrame, 0);
}

export function remainingSeconds(state, currentEta) {
  const perFrame = state.secondsPerFrame[rateKey(state.options)];
  if (perFrame == null || currentEta == null) return null;
  const queued = state.clips.filter((clip) => clip.status === "queued");
  return currentEta + queued.reduce((sum, clip) => sum + clip.frames * perFrame, 0);
}

export function overallFraction(clips) {
  const total = clips.reduce((sum, clip) => sum + clip.frames, 0);
  if (total === 0) return 0;
  return clips.reduce((sum, clip) => sum + clip.frames * clip.fraction, 0) / total;
}

export function projectName(state, fallback) {
  const dir = state.clips.find((clip) => clip.source_dir)?.source_dir;
  if (!dir) return fallback;
  return dir.split(/[\\/]/).filter(Boolean).pop() ?? fallback;
}

export function uncheckedFrames(clip) {
  return clip.analysis?.unchecked_frames ?? 0;
}

export function analyzedClips(state) {
  return state.clips.filter((clip) => clip.status === "analyzed" || (clip.status === "rendering" && clip.analysis));
}

// The clip Review shows: the selected one when it is analyzed, else the first analyzed one.
export function reviewClip(state) {
  const selected = selectedClip(state);
  return selected?.status === "analyzed" ? selected : (analyzedClips(state)[0] ?? null);
}

export function isReady(clip) {
  return clip.status === "analyzed" && (clip.analysis?.to_check ?? 0) === 0 && uncheckedFrames(clip) === 0;
}
