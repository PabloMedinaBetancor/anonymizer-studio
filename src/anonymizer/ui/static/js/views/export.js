import { h, nameNodes, svg, ICONS, replaceKeepingFocus } from "../dom.js";
import { toggle } from "../controls.js";
import { errorNotice } from "../api.js";
import { megabytes, clock, resolutionLabel, sizeParts, secondsText } from "../format.js";
import { uncheckedFrames } from "../state.js";
import {
  patternError, exampleName, positionAmong, exportableClips, isIncluded, blockedBy, renderable, planBody, totalBytes, fitsOnDisk, toCheckTotal,
  renderFraction, resultFiles, fileLabel, renderRequest, totalLowBytes,
} from "../export-logic.js";

const REPORT_FORMATS = ["pdf", "csv", "json"];

export function mountExport(root, { store, api, t }) {
  let plan = null;
  let planError = null;
  let planKey = "";
  let planTimer = 0;
  let planSerial = 0;
  let wasExporting = false;  // the plan is asked again when a render ends (what it answered during the render says "not possible" of the clips that were waiting)
  let wantedIds = [];  // the clips that will be written, in order: what {index} counts

  const clipList = h("div", { class: "clip-list" });
  const heading = h("span", { class: "section-title" }, t("export.clips"));
  const centerHead = h("div", { class: "center-head" }, h("span", { class: "head-label" }, t("export.files")), h("span", { class: "dim export-where" }));
  const where = centerHead.lastChild;
  const cards = h("div", { class: "export-cards" });
  const empty = h("div", { class: "empty-state" }, h("p", {}, t("export.empty")),
    h("button", { class: "btn", type: "button", onClick: () => store.dispatch({ type: "ui.tab", tab: "analyze" }) }, t("export.back_to_analyze")));

  // Options
  const destination = h("input", { class: "text-input", id: "export-destination", type: "text", spellcheck: "false", autocomplete: "off" });
  destination.addEventListener("input", () => store.dispatch({ type: "export.options", change: { destination: destination.value } }));
  const destinationNote = h("span", { class: "hint" });
  const destinationField = h("div", { class: "field" }, h("label", { class: "field-label", for: "export-destination" }, t("export.destination")), destination, destinationNote);
  const demoNote = h("p", { class: "note", hidden: true }, t("export.demo_note"));
  const pattern = h("input", { class: "text-input mono", id: "export-pattern", type: "text", spellcheck: "false", autocomplete: "off" });
  pattern.addEventListener("input", () => store.dispatch({ type: "export.options", change: { name_pattern: pattern.value } }));
  const example = h("span", { class: "hint mono" });
  const patternMessage = h("span", { class: "error", role: "alert", hidden: true });
  const chips = new Map(REPORT_FORMATS.map((format) => {
    const chip = h("button", { class: "chip", type: "button", "aria-pressed": "false", onClick: () => store.dispatch({ type: "export.report", format, value: chip.getAttribute("aria-pressed") !== "true" }) }, t(`export.report.${format}`));
    return [format, chip];
  }));
  const comparison = toggle({ label: t("export.comparison"), value: false, onChange: (value) => store.dispatch({ type: "export.options", change: { comparison: value } }) });
  const comparisonHint = h("span", { class: "hint" }, t("export.comparison.hint"));
  const summary = h("p", { class: "export-summary" });
  const space = h("p", { class: "hint export-space" });
  const toCheck = h("p", { class: "note", hidden: true });
  const problem = h("p", { class: "error", role: "alert", hidden: true });
  const renderButton = h("button", { class: "btn-primary", type: "button" });
  const stopButton = h("button", { class: "btn btn-danger", type: "button" }, t("analyze.stop"));
  const bar = h("div", { class: "bar" }, h("div", { class: "bar-fill" }));
  const progress = h("div", { class: "export-progress", hidden: true }, bar, stopButton);

  let starting = false;  // from the click until the job is on the page: a second click in that moment would only be answered "another job is running"
  renderButton.addEventListener("click", async () => {
    if (starting) return;
    const state = store.getState();
    const clips = renderable(state, plan);
    const ids = clips.map((clip) => clip.id);
    if (ids.length === 0) return;
    starting = true;
    renderButton.disabled = true;
    setTimeout(() => {  // the job event never came (a lost connection): the button must not stay dead
      starting = false;
      update(store.getState());
    }, 5000);
    try {
      const request = renderRequest(state, clips);
      await api.render(request.ids, request.options);
    } catch (failure) {
      starting = false;
      store.dispatch(errorNotice(failure));
      update(store.getState());
    }
  });
  stopButton.addEventListener("click", () => api.stop().catch((failure) => store.dispatch(errorNotice(failure))));

  const left = h("aside", { class: "side side-left", "aria-label": t("export.clips") }, h("div", { class: "side-head" }, heading), clipList);
  const center = h("section", { class: "center" }, centerHead, h("div", { class: "export-body" }, cards), empty);
  const right = h("aside", { class: "side side-right", "aria-label": t("export.options") },
    h("div", { class: "options" },
      h("span", { class: "section-title" }, t("export.options")),
      destinationField, demoNote,
      h("div", { class: "field" }, h("label", { class: "field-label", for: "export-pattern" }, t("export.pattern")), pattern, example, patternMessage),
      h("hr"),
      h("div", { class: "field" }, h("span", { class: "field-label" }, t("export.reports")), h("div", { class: "chips" }, ...chips.values())),
      comparison.el, comparisonHint),
    h("div", { class: "cta" }, toCheck, problem, summary, space, renderButton, progress));
  root.append(left, center, right);

  function schedulePlan(state, ids) {
    const key = JSON.stringify([ids, state.export.options, state.info?.capabilities, state.clips.map((c) => [c.id, c.analysis?.rev, c.status === "analyzed"])]);
    if (key === planKey) return;
    planKey = key;
    clearTimeout(planTimer);
    if (ids.length === 0 || patternError(state.export.options.name_pattern)) {
      plan = null;
      planError = null;
      return;
    }
    planTimer = setTimeout(async () => {
      planSerial += 1;
      const mine = planSerial;
      try {
        const answer = await api.renderPlan(ids, planBody(state, ids).options);
        if (mine !== planSerial) return;
        if (store.getState().job?.kind === "render") return;  // answered in the middle of a render: the clips still waiting are "blocked" in it
        plan = answer;
        planError = null;
      } catch (failure) {
        if (mine !== planSerial) return;
        plan = null;
        planError = errorNotice(failure);
      }
      update(store.getState());
    }, 250);
  }

  function blockedText(block) {
    return t(block.code, block.params);
  }

  function clipRow(state, clip, entry) {
    const included = isIncluded(state, clip);
    const block = blockedBy(clip, entry);
    const box = h("button", {
      class: "check", type: "button", role: "checkbox", "aria-checked": String(included && block == null), disabled: block != null || state.job != null,
      "aria-label": clip.name, onClick: () => store.dispatch({ type: "export.include", clip_id: clip.id, value: !included }),
    }, svg('<polyline points="4 12 10 18 20 6" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>', 12));
    const lines = [h("span", { class: "dim mono" }, clip.media === "photo" ? t("import.photo") : `${clock(clip.duration_s)} · ${resolutionLabel(clip.width, clip.height)}`)];
    if (block) lines.push(h("span", { class: "unchecked" }, blockedText(block)));
    else if ((clip.analysis?.to_check ?? 0) > 0) lines.push(h("span", { class: "notsure-text" }, t("export.clip_to_check", { count: clip.analysis.to_check })));
    return h("div", { class: "clip-card export-row", "data-clip": clip.id },
      box, h("span", { class: "clip-text" }, h("span", { class: "clip-name" }, nameNodes(clip.name)), ...lines));
  }

  function fileList(clip, state) {
    const demo = state.info?.mode === "demo";
    return h("ul", { class: "export-files" }, ...resultFiles(clip.render).map(([kind, path]) =>
      h("li", {}, h("span", { class: "pill" }, t(`export.file.${kind}`)),
        demo ? h("a", { class: "file-link", href: path, download: fileLabel(path) }, t("export.download", { name: fileLabel(path) })) : h("span", { class: "mono", title: path }, path))));
  }

  function card(state, clip, entry) {
    const block = blockedBy(clip, entry);
    const included = isIncluded(state, clip) && block == null;
    const parts = [h("div", { class: "card-line" }, h("span", { class: "clip-name" }, nameNodes(clip.name)), h("span", { class: "mono dim" }, clip.render?.phase === "done" && clip.render.outputs?.[0] ? fileLabel(clip.render.outputs[0]) : clip.render?.output ? fileLabel(clip.render.output) : entry?.output ? fileLabel(entry.output) : exampleName(state.export.options.name_pattern || "{name}", clip, positionAmong(wantedIds, clip))))];
    const r = clip.render;
    if (r && (r.phase === "render" || r.phase === "compare")) {
      parts.push(h("span", { class: "dim" }, t(r.phase === "render" ? "export.rendering" : "export.comparing", { percent: Math.round(r.fraction * 100) })),
        h("span", { class: "bar bar-small" }, h("span", { class: "bar-fill", style: `width:${Math.round(r.fraction * 100)}%` })));
    } else if (r?.phase === "queued") {
      parts.push(h("span", { class: "dim" }, t("analyze.queued")));
    } else if (r?.phase === "done") {
      parts.push(h("span", { class: "ok-text" }, t("export.done", { seconds: secondsText(r.seconds) })));
      if (r.comparison_stopped) parts.push(h("span", { class: "notsure-text" }, t("export.comparison_stopped")));
      parts.push(fileList(clip, state));
    } else if (block) {
      parts.push(h("span", { class: "unchecked" }, blockedText(block)));
    } else if (!included) {
      parts.push(h("span", { class: "dim" }, t("export.skipped")));
    } else {
      const facts = [];
      const size = entry?.estimate_bytes != null ? sizeParts(entry.estimate_low_bytes ?? entry.estimate_bytes, entry.estimate_bytes) : null;
      if (size) facts.push(size.key === "size" ? t("export.about", size.params) : t("export.about_range", size.params));
      if (entry?.exists) facts.push(t("export.renamed"));
      if (clip.media === "photo" && state.export.options.comparison) facts.push(t("comparison.photo_skipped"));
      parts.push(h("span", { class: "dim" }, facts.join(" · ")));
    }
    return h("article", { class: `export-card${included ? "" : " is-off"}` }, ...parts);
  }

  function update(state) {
    const clips = exportableClips(state);
    const wanted = clips.filter((clip) => isIncluded(state, clip) && uncheckedFrames(clip) === 0).map((clip) => clip.id);
    const exporting = state.job?.kind === "render";
    root.classList.toggle("is-idle", clips.length === 0);
    const o = state.export.options;
    const caps = state.info?.capabilities ?? {};
    const demo = state.info?.mode === "demo";
    wantedIds = wanted;
    if (wasExporting && !exporting) planKey = null;  // the render ended: what is there now (the names taken) is asked again, whatever was answered meanwhile
    wasExporting = exporting;
    if (!exporting) schedulePlan(state, wanted);
    const entries = new Map((plan?.clips ?? []).map((entry) => [entry.clip_id, entry]));

    heading.textContent = t("export.clips_heading", { count: clips.length });
    replaceKeepingFocus(clipList, clips.map((clip) => clipRow(state, clip, entries.get(clip.id))));
    replaceKeepingFocus(cards, clips.map((clip) => card(state, clip, entries.get(clip.id))));

    destinationField.hidden = demo;
    demoNote.hidden = !demo;
    if (document.activeElement !== destination) destination.value = o.destination;
    destination.placeholder = plan?.destination ?? t("export.destination.default");
    destinationNote.textContent = plan && plan.writable === false ? t("error.output.not_writable", { path: plan.unwritable?.[0] ?? plan.destination }) : t("export.destination.hint");
    destinationNote.classList.toggle("error", plan?.writable === false);
    // With the clips of several folders and no destination chosen, the outputs go next to each source: one folder to show would be wrong.
    where.textContent = plan && !demo ? (plan.destination ?? t("export.destination.default")) : "";
    where.title = where.textContent;  // cut with an ellipsis when the window is narrow: the whole path on hover
    if (document.activeElement !== pattern) pattern.value = o.name_pattern;
    const bad = patternError(o.name_pattern);
    patternMessage.hidden = bad == null;
    if (bad) patternMessage.textContent = t(bad);
    const first = clips.find((clip) => isIncluded(state, clip)) ?? clips[0];
    example.textContent = first && !bad ? t("export.example", { name: exampleName(o.name_pattern, first, positionAmong(wantedIds, first)) }) : "";

    for (const [format, chip] of chips) {
      const unavailable = format === "pdf" && !caps.pdf_report?.available;
      chip.disabled = unavailable || exporting;
      chip.title = unavailable ? t(caps.pdf_report?.reason ?? "report.pdf.no_browser", caps.pdf_report?.params ?? {}) : "";
      chip.setAttribute("aria-pressed", String(o.reports[format] && !unavailable));
    }
    const canCompare = Boolean(caps.comparison_video?.available);
    comparison.el.disabled = !canCompare || exporting;
    comparison.set(o.comparison && canCompare);
    comparisonHint.textContent = canCompare ? t("export.comparison.hint") : t("export.comparison.unavailable");

    const ready = renderable(state, plan);
    const bytes = totalBytes(plan, ready.map((c) => c.id));
    const total = plan ? sizeParts(totalLowBytes(plan, ready.map((c) => c.id)), bytes) : null;
    summary.textContent = ready.length === 0 ? "" : total?.key === "range"
      ? t("export.summary_range", { count: ready.length, ...total.params })
      : t("export.summary", { count: ready.length, size: total ? total.params.size : "-" });
    const fits = fitsOnDisk(plan, bytes);
    space.textContent = plan?.free_bytes != null && !demo ? t("export.free_space", { size: megabytes(plan.free_bytes) }) : "";
    const checks = toCheckTotal(ready);
    toCheck.hidden = checks === 0;
    if (checks > 0) toCheck.textContent = t("export.to_check", { count: checks });
    const message = !fits ? t("export.no_space") : planError ? t(planError.code, planError.params) : null;
    problem.hidden = message == null;
    if (message) problem.textContent = message;

    renderButton.hidden = exporting;
    progress.hidden = !exporting;
    const stopping = exporting && state.job.state === "stopping";  // a stop ends the clip being written first: say so
    stopButton.disabled = stopping;
    stopButton.textContent = t(stopping ? "analyze.stopping" : "analyze.stop");
    renderButton.textContent = t("export.render", { count: ready.length });
    if (state.job != null) starting = false;
    renderButton.disabled = ready.length === 0 || bad != null || !fits || state.job != null || plan == null || plan.writable === false || starting;
    if (exporting) bar.firstChild.style.width = `${Math.round(renderFraction(state.clips, o.comparison) * 100)}%`;
  }

  return { update };
}
