import { SCENE_DETECTIONS } from "./scene.js";

// Stand-in for the local server, used with ?mock to develop and screenshot the pages without the engine.
// Answers have the shapes of docs/UI-PROTOCOL.md.
const SUMMARIES = [
  { clip_id: "c1", name: "street-walk_0412.mp4", media: "video", source_dir: "/media/clips/Street shoot", duration_s: 108, frames: 3237, width: 1920, height: 1080, fps: 29.97, codec: "h264", has_audio: true, has_gps: true, size_bytes: 212e6 },
  { clip_id: "c2", name: "lobby-entrance.mov", media: "video", source_dir: "/media/clips/Street shoot", duration_s: 42, frames: 1260, width: 3840, height: 2160, fps: 30, codec: "h264", has_audio: true, has_gps: false, size_bytes: 480e6 },
  { clip_id: "c3", name: "parking-drone.mp4", media: "video", source_dir: "/media/clips/Street shoot", duration_s: 185, frames: 5550, width: 1920, height: 1080, fps: 30, codec: "h264", has_audio: false, has_gps: true, size_bytes: 390e6 },
].map((c) => ({ ...c, status: "imported", analysis: null, thumb_url: null, source_url: null }));

const ITEMS = [
  ["face", 0.02, 0.17, "blur", 0.97], ["plate", 0.1, 0.4, "blur", 0.94], ["face", 0.17, 0.25, "check", 0.58],
  ["face", 0.36, 0.5, "blur", 0.91], ["plate", 0.5, 0.57, "blur", 0.88], ["face", 0.57, 0.7, "blur", 0.86],
  ["plate", 0.6, 0.7, "check", 0.41], ["face", 0.8, 0.97, "blur", 0.95], ["plate", 0.88, 0.95, "blur", 0.9],
];

const FOLDERS = {
  "/media/clips": { parent: "/media", entries: [{ name: "Street shoot", kind: "dir", items: 3 }] },
  "/media/clips/Street shoot": {
    parent: "/media/clips",
    entries: [...SUMMARIES.map((c) => ({ name: c.name, kind: "video", size_bytes: c.size_bytes })), { name: "IMG_0001.HEIC", kind: "unsupported", size_bytes: 3.1e6 }],
  },
};

function itemsFor(clip) {
  return ITEMS.map(([kind, from, to, verdict, score], index) => ({
    item_id: index, kind, judgement: { verdict, reason: `${kind}.reason.${verdict === "blur" ? (kind === "face" ? "verified" : "on_vehicle_verified") : kind === "face" ? "low_verify" : "unconfirmed"}`, params: {} },
    first_frame: Math.round(from * clip.frames), last_frame: Math.round(to * clip.frames), max_score: score, blur: true, reviewed: false, manual: false, thumb_url: null,
    found_by: index === 2 || index === 6 ? "second" : "light", second_score: index === 2 ? 0.62 : index === 6 ? 0.55 : null, decided_by: "engine",
  }));
}

const SAMPLES = [
  { sample_id: "street-1", name: "street-1.mp4", media: "video", duration_s: 14, licence: "CC BY 4.0", author: "Example Author", source_url: "https://example.org/street-1" },
  { sample_id: "plate-1", name: "plate-1.jpg", media: "photo", duration_s: 0, licence: "CC0", author: "Example Author", source_url: "https://example.org/plate-1" },
];

export function createMockApi({ tick = 250, step = 0.012, demo = false } = {}) {
  // Decisions made in Review are kept here, as the server keeps them.
  const decided = new Map();
  let rev = 0;
  const store = (clipId) => {
    if (!decided.has(clipId)) decided.set(clipId, itemsFor(SUMMARIES.find((c) => c.clip_id === clipId)));
    return decided.get(clipId);
  };
  const toCheck = (clipId) => store(clipId).filter((i) => i.judgement.verdict === "check" && !i.reviewed).length;
  let listener = () => {};
  let timer = null;
  let paused = false;
  let queue = [];
  let progress = 0;
  let imported = [];
  const emit = (message) => listener({ v: 1, ...message });

  function finish(status) {
    clearInterval(timer);
    timer = null;
    emit({ type: "job.finished", job_id: "j1", status, summary: {} });
  }

  function advance() {
    if (paused) return;
    const clip = SUMMARIES.find((c) => c.clip_id === queue[0]);
    if (progress === 0) emit({ type: "clip.started", clip_id: clip.clip_id, phase: "analyze" });
    progress = Math.min(1, progress + step);
    const frame = Math.round(progress * clip.frames);
    const shift = Math.round(Math.sin(frame / 40) * 8);
    const live = SCENE_DETECTIONS.map((d) => ({ ...d, box: [d.box[0] + shift, d.box[1], d.box[2] + shift, d.box[3]] }));
    const second = progress >= 0.8 ? { stage: "second", zones: { done: Math.round(((progress - 0.8) / 0.2) * 14), total: 14 } } : { stage: "first" };
    emit({ type: "clip.progress", clip_id: clip.clip_id, phase: "analyze", fraction: progress, frame, frames: clip.frames, eta_s: ((1 - progress) / step) * (tick / 1000), live, ...second });
    if (progress < 1) return;
    const items = store(clip.clip_id);
    for (const item of items) emit({ type: "clip.item", clip_id: clip.clip_id, item: { ...item } });
    emit({ type: "clip.analyzed", clip_id: clip.clip_id, items: items.length, to_check: items.filter((i) => i.judgement.verdict === "check").length, unchecked_frames: 0, rev: 0, analysis_url: "" });
    queue.shift();
    progress = 0;
    if (queue.length === 0) finish("done");
  }

  return {
    state: async () => ({
      v: 1, mode: demo ? "demo" : "local", languages: ["en", "es"], slowdown: { maximum: 4.6, careful: 3.1 },
      demo_limits: demo ? { max_photos: 10, max_seconds: 20, max_height: 720, max_upload_mb: 100 } : null,
      capabilities: { pdf_report: { available: true }, comparison_video: { available: true }, browse: !demo, upload: true },
      clips: imported, job: null, seq: 0,
    }),
    settings: async () => ({ language: "en", faces: true, plates: true, thoroughness: "standard", detection: "careful", gap_fill: "camera", blur: "gaussian", strength: 1.0, extra_margin: 0.15, remove_gps: true }),
    putSettings: async (change) => change,
    browse: async (path) => {
      const folder = FOLDERS[path ?? "/media/clips"];
      if (!folder) throw new Error(`no such folder: ${path}`);
      return { path: path ?? "/media/clips", parent: folder.parent, roots: [], entries: folder.entries, truncated: false };
    },
    importPaths: async () => {
      imported = SUMMARIES;
      const refused = (name, code) => ({ path: `/media/clips/Street shoot/${name}`, code, params: { name }, detail: "" });
      return {
        clips: SUMMARIES,
        rejected: [
          refused("IMG_0001.HEIC", "error.media.heic"), refused("anim.gif", "error.media.gif"), refused("empty.jpg", "error.media.empty"),
          refused("corrupt.jpg", "error.media.unreadable_photo"), refused("notvideo.mp4", "error.media.unreadable_video"), refused("IMG_0002.CR2", "error.media.raw"),
        ],
      };
    },
    upload: async () => ({ clips: [], rejected: [] }),
    analysis: async (clipId) => ({ clip_id: clipId, rev, items: store(clipId).map((i) => ({ ...i })) }),
    async analyze(ids) {
      queue = [...ids];
      progress = 0;
      paused = false;
      const clips = SUMMARIES.filter((c) => ids.includes(c.clip_id));
      emit({ type: "job.started", job_id: "j1", kind: "analyze", clips });
      timer = setInterval(advance, tick);
      return { job_id: "j1", clips: ids, estimate_s: clips.reduce((sum, c) => sum + c.frames * 0.02, 0) };
    },
    samples: async () => SAMPLES,
    importSamples: async () => {
      imported = SUMMARIES;
      return { clips: SUMMARIES, rejected: [] };
    },
    async renderPlan(ids, options) {
      const pattern = options.name_pattern ?? "{name}_anonymized";
      const destination = options.destination ?? "/media/clips/Street shoot/anonymized";
      const clips = ids.map((id, index) => {
        const clip = SUMMARIES.find((c) => c.clip_id === id);
        const stem = clip.name.replace(/\.[^.]+$/, "");
        const name = pattern.split("{name}").join(stem).split("{index}").join(String(index + 1));
        return { clip_id: id, output: `${destination}/${name}.mp4`, exists: id === "c2", estimate_bytes: Math.round(clip.size_bytes * 1.6), estimate_low_bytes: Math.round(clip.size_bytes * 0.5), blocked: null };
      });
      return { destination, writable: true, free_bytes: 52e9, clips, total_bytes: clips.reduce((sum, c) => sum + c.estimate_bytes, 0), total_low_bytes: clips.reduce((sum, c) => sum + c.estimate_low_bytes, 0) };
    },
    async render(ids, options) {
      const clips = SUMMARIES.filter((c) => ids.includes(c.clip_id));
      emit({ type: "job.started", job_id: "j2", kind: "render", clips });
      const plan = await this.renderPlan(ids, options);
      let at = 0;
      let done = 0;
      let fraction = 0;
      timer = setInterval(() => {
        const clip = clips[at];
        if (fraction === 0) emit({ type: "clip.started", clip_id: clip.clip_id, phase: "render" });
        fraction = Math.min(1, fraction + step * 3);
        emit({ type: "clip.progress", clip_id: clip.clip_id, phase: "render", fraction, frame: Math.round(fraction * clip.frames), frames: clip.frames, eta_s: null });
        if (fraction < 1) return;
        const out = plan.clips[at].output;
        const link = (name) => (demo ? `/api/downloads/f${at + 1}/${name.split("/").pop()}` : name);
        const base = out.replace(/\.mp4$/, "");
        emit({
          type: "clip.rendered", clip_id: clip.clip_id, outputs: [link(out)], seconds: 4 + at,
          report: Object.fromEntries(["pdf", "csv", "json"].filter((k) => options.reports?.[k]).map((k) => [k, link(`${base}.report.${k}`)])),
          ...(options.comparison ? { comparison: link(`${base}.compare.mp4`) } : {}),
        });
        fraction = 0;
        at += 1;
        done += 1;
        if (done === clips.length) {
          clearInterval(timer);
          timer = null;
          emit({ type: "job.finished", job_id: "j2", status: "done", summary: { rendered: done } });
        }
      }, tick);
      return { job_id: "j2", clips: ids, outputs: {} };
    },
    pause: async () => { paused = true; emit({ type: "job.state", state: "paused" }); },
    resume: async () => { paused = false; emit({ type: "job.state", state: "running" }); },
    stop: async () => finish("stopped"),
    previewUrl: () => null,
    boxes: async (clipId, from, to, rev) => ({ rev, frame_from: from, frame_to: to, fps: 30, frames: {} }),
    async setItem(clipId, itemId, blur) {
      const item = store(clipId).find((i) => i.item_id === itemId);
      Object.assign(item, { blur, reviewed: true });
      return { item: { ...item }, rev: ++rev, to_check: toCheck(clipId) };
    },
    async setDoubtful(clipId, blur) {
      const items = store(clipId).filter((i) => i.judgement.verdict === "check");
      for (const item of items) Object.assign(item, { blur, reviewed: true });
      return { items: items.map((item) => ({ ...item })), rev: ++rev, to_check: toCheck(clipId) };
    },
    async addArea(clipId, { frame_from, frame_to, box }) {
      const items = store(clipId);
      const item = {
        item_id: 100 + items.length, kind: "manual", judgement: { verdict: "blur", reason: "manual.reason.added", params: {} },
        first_frame: frame_from, last_frame: frame_to, max_score: 1, blur: true, reviewed: true, manual: true, thumb_url: null, box,
      };
      items.push(item);
      return { item: { ...item }, boxes: {}, rev: ++rev };
    },
    async removeArea(clipId, itemId) {
      const items = store(clipId);
      items.splice(items.findIndex((i) => i.item_id === itemId), 1);
      return { ok: true, rev: ++rev, to_check: toCheck(clipId) };
    },
    onEvent(handler) {
      listener = handler;
      return () => { listener = () => {}; };
    },
  };
}
