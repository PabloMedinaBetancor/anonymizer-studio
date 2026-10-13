import { h, nameNodes, svg, ICONS, replaceKeepingFocus } from "../dom.js";
import { toggle, segmented, select, slider } from "../controls.js";
import { createStage } from "../stage.js";
import { createPlayer, browserCanPlay } from "../player.js";
import { errorNotice } from "../api.js";
import { openExplorer } from "../explorer.js";
import { openSamples } from "../samples.js";
import { askConfirm } from "../confirm-dialog.js";
import { clock, fixed, timecode, durationParts, resolutionLabel, megabytes } from "../format.js";
import { analyzableClips, clipsToAnalyze, refusalNotices, sortDropped, analyzeOptions, estimateSeconds, rateKey, selectedClip } from "../state.js";

export function mountImport(root, { store, api, t }) {
  const stage = createStage(t);
  const timeLabel = h("span", { class: "timecode" });
  const scrub = h("input", { type: "range", min: 0, max: 1000, value: 0, class: "scrub", "aria-label": t("player.seek") });
  const playButton = h("button", { class: "btn-play", type: "button", "aria-label": t("player.play") }, svg(ICONS.play));
  let mode = "simulated";
  let current = null;
  const player = createPlayer(({ time, playing }) => {
    if (!current) return;
    timeLabel.replaceChildren(
      timecode(time, current.fps),
      h("span", { class: "dim" }, ` / ${timecode(current.duration_s, current.fps)}`),
    );
    scrub.value = String(Math.round((time / current.duration_s) * 1000));
    playButton.replaceChildren(svg(playing ? ICONS.pause : ICONS.play));
    playButton.setAttribute("aria-label", t(playing ? "player.pause" : "player.play"));
    if (mode === "frames") stage.showImage(api.previewUrl(current.id, Math.round(time * current.fps)));
  });
  playButton.addEventListener("click", () => player.toggle());
  scrub.addEventListener("input", () => {
    if (current) player.seek((Number(scrub.value) / 1000) * current.duration_s);
  });

  const mediaHeading = h("span", { class: "section-title" });
  const clipList = h("div", { class: "clip-list" });
  const dropZone = h("div", { class: "drop-zone" }, t("import.drop"));
  const importButton = h("button", { class: "btn btn-small", type: "button" }, svg(ICONS.plus, 13), t("import.button"));
  importButton.addEventListener("click", () => {
    if (store.getState().info?.mode === "demo") openSamples({ api, t, onPick: (ids) => importSamples(ids) });
    else openExplorer({ api, t, onPick: (paths) => importPaths(paths) });
  });
  const importLabel = importButton.lastChild;
  const limits = h("p", { class: "hint demo-limits", hidden: true });
  const idleNote = h("p", { class: "hint idle-note", hidden: true });

  // `collect`: a list to put the refusals in, to be told together by the caller (files that go up one at a time are one import to the person)
  function addResult(result, collect = null) {
    store.dispatch({ type: "clips.added", clips: result.clips });
    if (collect) collect.push(...result.rejected);
    else tellRefusals(result.rejected);
  }

  function tellRefusals(rejected) {
    for (const notice of refusalNotices(rejected)) store.dispatch({ type: "job.warning", ...notice });
  }

  async function importSamples(ids) {
    try {
      addResult(await api.importSamples(ids));
    } catch (failure) {
      store.dispatch(errorNotice(failure));
    }
  }

  // Reading a folder of large photos takes a while (a quarter of a second each) and the answer is one request: say so meanwhile.
  let importing = 0;
  function busyImporting(on) {
    importing += on ? 1 : -1;
    update(store.getState());
  }

  async function importPaths(paths) {
    busyImporting(true);
    try {
      addResult(await api.importPaths(paths));
    } catch (failure) {
      store.dispatch(errorNotice(failure));
    } finally {
      busyImporting(false);
    }
  }

  // The whole Import screen takes a drop (the box at the bottom of the list only shows where): a file dropped anywhere else on the page
  // used to make the browser open it and leave the Studio.
  const carriesFiles = (event) => [...(event.dataTransfer?.types ?? [])].includes("Files");
  root.addEventListener("dragover", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    event.stopPropagation();  // the window's own handler says "no drop here": this screen says yes
    event.dataTransfer.dropEffect = "copy";
    dropZone.classList.add("over");
  });
  root.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget || !root.contains(event.relatedTarget)) dropZone.classList.remove("over");
  });
  // What was dropped, as files: a dropped folder is walked (its hidden files and the folder the Studio writes into are left out, as the server's own
  // folder import does, and the hidden ones are counted and said), and what is in it is sorted before anything is sent (`sortDropped`: the
  // files the server would only refuse are not uploaded). The entries have to be taken while the drop event is being handled, before the
  // first wait.
  async function droppedFiles(transfer) {
    const entries = [...(transfer.items ?? [])].filter((item) => item.kind === "file").map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
    if (entries.length === 0) return { files: [...transfer.files], refused: [] };
    const found = [];
    let hidden = 0;
    // The Studio's lists of what it takes come with its state: a drop on a page whose state has not arrived yet (a slow computer, the first moments)
    // asks for it, instead of sending every file of the folder for the server to refuse.
    const formats = store.getState().info?.formats ?? (await api.state().then((snapshot) => snapshot.formats ?? null, () => null));
    const readAll = (reader) => new Promise((resolve, reject) => {
      const all = [];
      const more = () => reader.readEntries((batch) => (batch.length === 0 ? resolve(all) : (all.push(...batch), more())), reject);
      more();
    });
    async function walk(entry, inFolder) {
      if (inFolder && entry.name.startsWith(".")) {
        hidden += 1;
        return;
      }
      if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        found.push({ file, name: file.name, fromFolder: inFolder });
      } else if (entry.isDirectory && !(inFolder && formats?.output_folder === entry.name)) {
        for (const child of await readAll(entry.createReader())) await walk(child, true);
      }
    }
    for (const entry of entries) await walk(entry, false);
    const { upload, refused } = sortDropped(found, formats);
    if (hidden > 0) refused.push({ path: "", code: "warning.import.hidden", params: { count: hidden } });
    return { files: upload.map((entry) => entry.file), refused };
  }

  root.addEventListener("drop", async (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    dropZone.classList.remove("over");
    let dropped;
    try {
      dropped = await droppedFiles(event.dataTransfer);
    } catch (failure) {
      store.dispatch(errorNotice(failure));
      return;
    }
    await uploadFiles(dropped.files, dropped.refused);
  });

  async function uploadFiles(files, alreadyRefused = []) {
    busyImporting(true);
    const refused = [...alreadyRefused];
    try {
      for (const file of files) {
        try {
          await file.slice(0, 1).arrayBuffer();  // a file the browser cannot read (no permission) is refused by name, not as "the engine did not answer"
        } catch {
          refused.push({ path: file.name, code: "error.media.denied", params: { name: file.name } });
          continue;
        }
        try {
          addResult(await api.upload(file), refused);
        } catch (failure) {
          store.dispatch(errorNotice(failure));
        }
      }
    } finally {
      tellRefusals(refused);
      busyImporting(false);
    }
  }

  // In the hosted demo there is no folder browser: a click on the box under the list opens the browser's own file chooser (a tablet or a phone
  // cannot drag a file from anywhere), and the files go up as a drop does.
  const chooser = h("input", { type: "file", multiple: true, accept: "image/*,video/*", hidden: true, tabindex: "-1", "aria-hidden": "true" });
  chooser.addEventListener("change", async () => {
    const files = [...chooser.files];
    chooser.value = "";
    if (files.length > 0) await uploadFiles(files);
  });
  const chooseFiles = () => {
    if (store.getState().info?.mode === "demo" && importing === 0) chooser.click();
  };
  dropZone.addEventListener("click", chooseFiles);
  dropZone.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    chooseFiles();
  });

  const sourceTitle = h("span", { class: "source-title" });
  const controls = h("div", { class: "controls" },
    h("button", { class: "btn-icon", type: "button", "aria-label": t("player.prev_frame"), onClick: () => player.step(-1) }, svg(ICONS.prev)),
    playButton,
    h("button", { class: "btn-icon", type: "button", "aria-label": t("player.next_frame"), onClick: () => player.step(1) }, svg(ICONS.next)),
    timeLabel, scrub);
  const meta = h("footer", { class: "meta-bar" });
  // With nothing imported the middle of the screen says what to do instead of showing a stand-in picture with dead controls.
  const emptyBody = h("p", {}, t("import.empty_body"));
  const emptyButton = h("button", { class: "btn-primary", type: "button", onClick: () => importButton.click() });
  const emptyState = h("div", { class: "empty-state import-empty" }, svg(ICONS.film, 44),
    h("h2", {}, t("import.empty_title")), emptyBody, emptyButton);

  // Faces and plates cannot both be off (the server refuses it): the last one stays on, and the person is told why.
  function detect(key, other) {
    return (value) => {
      if (!value && !store.getState().options[other]) {
        store.dispatch({ type: "job.warning", code: "options.one_required", params: {} });
        return;
      }
      store.dispatch({ type: "ui.options", change: { [key]: value } });
    };
  }
  const faces = toggle({ label: t("options.faces"), swatch: "face", value: true, onChange: detect("faces", "plates") });
  const plates = toggle({ label: t("options.plates"), swatch: "plate", value: true, onChange: detect("plates", "faces") });
  const DETECTIONS = ["fast", "careful"];
  const detection = segmented({
    label: t("options.detection"),
    options: DETECTIONS.map((value) => ({ value, label: t(`options.detection.${value}`) })),
    value: "careful",
    onChange: (v) => store.dispatch({ type: "ui.options", change: { detection: v } }),
  });
  const detectionHint = h("span", { class: "hint" });
  const gaps = select({
    id: "gap-fill",
    value: "camera",
    options: [
      { value: "camera", label: t("options.gap.camera") },
      { value: "straight", label: t("options.gap.straight") },
      { value: "hold", label: t("options.gap.hold") },
    ],
    onChange: (v) => store.dispatch({ type: "ui.options", change: { gap_fill: v } }),
  });
  const blurStyle = select({
    id: "blur-style",
    value: "gaussian",
    options: [
      { value: "gaussian", label: t("options.blur.gaussian") },
      { value: "pixelate", label: t("options.blur.pixelate") },
      { value: "solid", label: t("options.blur.solid") },
    ],
    onChange: (v) => store.dispatch({ type: "ui.options", change: { blur: v } }),
  });
  const strength = slider({ id: "strength", min: 100, max: 200, value: 100, onChange: (v) => store.dispatch({ type: "ui.options", change: { strength: v / 100 } }) });
  const marginSlider = slider({ id: "margin", min: 0, max: 50, value: 15, onChange: (v) => store.dispatch({ type: "ui.options", change: { extra_margin: v / 100 } }) });
  const marginLabel = h("label", { class: "field-label", for: "margin" });
  const strengthLabel = h("label", { class: "field-label", for: "strength" });
  const gps = toggle({ label: t("options.remove_gps"), value: true, onChange: (v) => store.dispatch({ type: "ui.options", change: { remove_gps: v } }) });

  const analyzeButton = h("button", { class: "btn-primary", type: "button" });
  const estimate = h("span", { class: "estimate" });
  // From the click until the job is on the page a second click would only be answered "another job is running" (see the Export button).
  let starting = false;
  const releaseStarting = () => {
    starting = false;
    update(store.getState());
  };
  analyzeButton.addEventListener("click", async () => {
    if (starting) return;
    starting = true;
    analyzeButton.disabled = true;
    setTimeout(releaseStarting, 5000);  // the job event never came (a lost connection): the button must not stay dead
    if (!(await startAnalysis())) {
      releaseStarting();
      // The button was dead while the question stood and gave up the focus: after Cancel the person is back on it, not on the page's body.
      if (document.activeElement === document.body || document.activeElement === null) analyzeButton.focus({ preventScroll: true });
    }
  });

  // Whether the analysis was started.
  async function startAnalysis() {
    const state = store.getState();
    const clips = clipsToAnalyze(state);
    const level = rateKey(state.options);
    try {
      const ids = clips.map((c) => c.id);
      let started;
      try {
        started = await api.analyze(ids, analyzeOptions(state.options));
      } catch (failure) {
        // Other settings would replace the analysis of clips the person has reviewed, with their decisions: ask first.
        if (failure.code !== "error.analysis.discards") throw failure;
        const yes = await askConfirm({ t, title: t("import.discard_title"), body: t(failure.code, failure.params), confirm: t("import.discard_confirm") });
        if (!yes) return false;
        started = await api.analyze(ids, analyzeOptions(state.options), true);
      }
      const frames = clips.reduce((sum, c) => sum + c.frames, 0);
      if (started.estimate_s != null && frames > 0) store.dispatch({ type: "estimate", key: level, seconds_per_frame: started.estimate_s / frames });
      return true;
    } catch (failure) {
      store.dispatch(errorNotice(failure));
      return false;
    }
  }

  const label = (text, forId) => h("label", { class: "field-label", for: forId }, text);
  const row = (...children) => h("div", { class: "row" }, ...children);

  const left = h("aside", { class: "side side-left", "aria-label": t("import.media") },
    h("div", { class: "side-head" }, mediaHeading, importButton), clipList, limits, dropZone, idleNote);
  const center = h("section", { class: "center", "aria-label": t("import.viewer") },
    h("div", { class: "center-head" }, sourceTitle, h("span", { class: "dim" }, t("player.fit"))),
    h("div", { class: "center-body" }, h("div", { class: "stage-wrap" }, stage.el),
      controls,
),
    meta, emptyState);
  const right = h("aside", { class: "side side-right", "aria-label": t("options.title") },
    h("div", { class: "options" },
      h("span", { class: "section-title" }, t("options.detect")),
      faces.el, plates.el,
      h("div", { class: "field" }, h("span", { class: "dim" }, t("options.detection")), detection.el, detectionHint),
      h("div", { class: "field" }, label(t("options.gap"), "gap-fill"), gaps.el, h("span", { class: "hint" }, t("options.gap.hint"))),
      h("hr"),
      h("span", { class: "section-title" }, t("options.blur")),
      row(label(t("options.style"), "blur-style"), blurStyle.el),
      row(strengthLabel, strength.el),
      row(marginLabel, marginSlider.el),
      h("hr"),
      h("span", { class: "section-title" }, t("options.output")),
      row(h("span", { class: "dim" }, t("options.video")), h("span", {}, t("options.video.match"))),
      row(h("span", { class: "dim" }, t("options.audio")), h("span", {}, t("options.audio.copy"))),
      gps.el),
    h("div", { class: "cta" }, analyzeButton, estimate));
  root.append(left, center, right, chooser);

  let shownClipId = null;

  async function removeClip(clip) {
    try {
      await api.removeClip(clip.id);
      store.dispatch({ type: "clip.removed", clip_id: clip.id });
    } catch (failure) {
      store.dispatch(errorNotice(failure));
    }
  }

  function renderClips(state) {
    mediaHeading.textContent = t("import.media_heading", { count: state.clips.length });
    const rows = state.clips.map((clip) =>
        h("div", { class: "clip-row" },
          h("button", { class: "clip-card", type: "button", "aria-pressed": String(clip.id === state.selectedId), onClick: () => store.dispatch({ type: "ui.select", clip_id: clip.id }) },
            h("span", { class: "thumb" }, clip.thumb_url ? h("img", { src: clip.thumb_url, alt: "" }) : svg(ICONS.film, 22)),
            h("span", { class: "clip-text" },
              h("span", { class: "clip-name" }, nameNodes(clip.name)),
              h("span", { class: "mono dim" }, clip.media === "photo" ? t("import.photo") : `${clock(clip.duration_s)} \u00b7 ${resolutionLabel(clip.width, clip.height)}`))),
          // Only the list is changed: the file is not touched. Not while a job holds the clip (the server refuses it as well). The second click of a
          // double click is not a second removal: the list has closed up by then and it would fall on the next clip's cross.
          h("button", { class: "btn-icon clip-remove", type: "button", disabled: state.job != null, "aria-label": t("import.remove", { name: clip.name }), title: t("import.remove", { name: clip.name }), onClick: (event) => { if (event.detail < 2) removeClip(clip); } }, svg(ICONS.close, 12))));
    if (state.clips.length === 0) rows.push(h("p", { class: "empty" }, t(state.info?.mode === "demo" ? "import.empty.demo" : "import.empty")));
    replaceKeepingFocus(clipList, rows);
  }

  function renderMeta(clip) {
    if (!clip) {
      meta.replaceChildren();
      return;
    }
    const video = clip.media === "video";
    const parts = [
      `${clip.width}\u00d7${clip.height}`,
      video ? `${fixed(Number(clip.fps), 2)} fps` : null,
      clip.codec,
      video ? t(clip.has_audio ? "meta.audio_yes" : "meta.audio_no") : null,
      megabytes(clip.size_bytes),
      t(clip.has_gps ? "meta.gps_yes" : "meta.gps_no"),
    ].filter(Boolean);
    meta.replaceChildren(...parts.map((p) => h("span", {}, p)));
  }

  function showClip(clip) {
    current = clip;
    stage.setClip(clip);
    controls.hidden = clip.media === "photo";
    if (clip.media === "photo") mode = "photo";
    else if (!clip.source_url) mode = "simulated";
    else mode = browserCanPlay(clip) ? "video" : "frames";
    playButton.disabled = mode === "frames";
    if (mode === "video") {
      stage.showVideo();
      player.attach(clip, stage.video);
    } else {
      player.attach(clip, null);
      stage.showImage(mode === "photo" ? clip.source_url : mode === "frames" ? api.previewUrl(clip.id, 0) : null);
    }
  }

  function update(state) {
    renderClips(state);
    const demo = state.info?.mode === "demo";
    importLabel.textContent = t(importing > 0 ? "import.importing" : demo ? "import.samples" : "import.button");
    importButton.disabled = importing > 0;
    limits.hidden = !demo || !state.info.demo_limits;
    if (demo && state.info.demo_limits) {
      const l = state.info.demo_limits;
      limits.textContent = t("import.demo_limits", { photos: l.max_photos, seconds: l.max_seconds, height: l.max_height, mb: l.max_upload_mb });
    }
    center.classList.toggle("is-idle", state.clips.length === 0);
    emptyBody.textContent = t(demo ? "import.empty_body.demo" : "import.empty_body");
    dropZone.textContent = t(demo ? "import.drop.demo" : "import.drop");
    dropZone.classList.toggle("is-button", demo);
    if (demo) dropZone.setAttribute("role", "button");
    else dropZone.removeAttribute("role");
    if (demo) dropZone.setAttribute("tabindex", "0");
    else dropZone.removeAttribute("tabindex");
    emptyButton.textContent = importLabel.textContent;
    emptyButton.disabled = importing > 0;
    idleNote.hidden = demo || !state.info?.idle;
    if (!idleNote.hidden) idleNote.textContent = t("import.idle_note", state.info.idle);
    const clip = selectedClip(state);
    sourceTitle.replaceChildren(h("span", { class: "dim" }, `${t("import.source")} \u00b7 `), clip?.name ?? "");
    sourceTitle.title = clip?.name ?? "";
    if (clip && clip.id !== shownClipId) {
      shownClipId = clip.id;
      showClip(clip);
    }
    renderMeta(clip);

    const o = state.options;
    faces.set(o.faces);
    plates.set(o.plates);
    const costs = state.info?.costs ?? {};
    detection.set(o.detection);
    for (const value of DETECTIONS) detection.setLabel(value, costs[value] ? t(`options.detection.${value}.slower`, { factor: costs[value] }) : t(`options.detection.${value}`));
    detectionHint.textContent = t(`options.detection.${o.detection}.hint`);
    gaps.set(o.gap_fill);
    blurStyle.set(o.blur);
    strength.set(Math.round(o.strength * 100));
    strengthLabel.textContent = t("options.strength", { value: fixed(o.strength, 2) }); // the slider's own value, as the margin has its own
    marginSlider.set(Math.round(o.extra_margin * 100));
    marginLabel.textContent = t("options.margin", { percent: Math.round(o.extra_margin * 100) });
    gps.set(o.remove_gps);

    const clips = clipsToAnalyze(state);
    const busy = state.job != null;
    // "Analyze 0 clips" said nothing a disabled button does not: with no project, or with the clips busy, it is just "Analyze".
    analyzeButton.textContent = clips.length > 0 ? t("analyze.cta", { count: clips.length }) : analyzableClips(state).length > 0 ? t("analyze.cta.none") : t("nav.analyze");
    if (busy) starting = false;
    analyzeButton.disabled = busy || clips.length === 0 || !(o.faces || o.plates) || starting;
    const parts = durationParts(estimateSeconds(clips, state));
    estimate.textContent = parts ? t("analyze.estimate", { time: t(parts.key, parts.params) }) : "";
  }

  return { update, destroy: () => player.stop() };
}
