import { h, nameNodes, reconcile } from "../dom.js";
import { createStage } from "../stage.js";
import { errorNotice } from "../api.js";
import { packRows, isNotSure } from "../lanes.js";
import { itemCounts, overallFraction, remainingSeconds, selectedClip, uncheckedFrames } from "../state.js";
import { durationParts, frameLabel, liveClock, percentText } from "../format.js";

export function mountAnalyze(root, { store, api, t, openReview }) {
  const stage = createStage(t);
  const clipList = h("div", { class: "clip-list" });
  const heading = h("span", { class: "section-title" });
  const title = h("span", { class: "source-title" });
  const frameCount = h("span", { class: "dim" });
  const badge = h("span", { class: "badge-live mono" });
  const foundSummary = h("span", { class: "found-summary" });
  const lanes = h("div", { class: "found-lanes" });
  const progressRows = h("div", { class: "progress-rows" });
  const bar = h("div", { class: "bar" }, h("div", { class: "bar-fill" }));
  const secondText = h("span", { class: "dim" });
  const secondBar = h("div", { class: "bar bar-second" }, h("div", { class: "bar-fill" }));
  const second = h("div", { class: "second-look", hidden: true }, secondText, secondBar);
  const settings = h("p", { class: "settings-used" });
  const pause = h("button", { class: "btn btn-wide", type: "button" });
  const stop = h("button", { class: "btn btn-wide btn-danger", type: "button" }, t("analyze.stop"));
  const empty = h("div", { class: "empty-state" }, h("p", {}, t("analyze.idle")),
    h("button", { class: "btn", type: "button", onClick: () => store.dispatch({ type: "ui.tab", tab: "import" }) }, t("analyze.back_to_import")));

  const send = (call) => call().catch((failure) => store.dispatch(errorNotice(failure)));
  pause.addEventListener("click", () => send(() => (store.getState().job?.state === "paused" ? api.resume() : api.pause())));
  stop.addEventListener("click", () => send(() => api.stop()));

  const left = h("aside", { class: "side side-left" }, h("div", { class: "side-head" }, heading), clipList);
  const center = h("section", { class: "center" },
    h("div", { class: "center-head" }, title, frameCount),
    h("div", { class: "center-body" }, h("div", { class: "stage-wrap" }, stage.el)),
    h("section", { class: "found", "aria-label": t("analyze.found") },
      h("div", { class: "found-head" }, h("span", { class: "section-title" }, t("analyze.found")), foundSummary), lanes),
    empty);
  const right = h("aside", { class: "side side-right" },
    h("div", { class: "options" },
      h("span", { class: "section-title" }, t("analyze.progress")), progressRows, bar, second,
      h("div", { class: "button-pair" }, pause, stop),
      h("hr"), h("span", { class: "section-title" }, t("analyze.settings_used")), settings),
    h("div", { class: "cta" }, h("p", { class: "note" }, t("analyze.note"))));
  stage.el.append(badge);  // in the picture's corner, not the corner of the wrap that is larger than it
  root.append(left, center, right);

  function row(name, value) {
    return h("div", { class: "row" }, h("span", { class: "dim" }, name), h("span", {}, value));
  }

  function etaText(seconds) {
    const parts = durationParts(seconds);
    if (parts?.key === "duration.under_minute") return t("analyze.left_under_minute");
    return parts ? t("analyze.about_left", { time: t(parts.key, parts.params) }) : t("analyze.estimating");
  }

  // One entry per clip for reconcile: its card is made again only when something it shows changed, so the card under the pointer is the one
  // the pointer went down on (a progress event of another clip does not touch it).
  function clipEntry(clip) {
    const lines = [];
    if (clip.status === "analyzing") lines.push(etaText(clip.eta_s));
    else if (clip.status === "queued") lines.push(t("analyze.queued"));
    else if (clip.status === "analyzed") lines.push(t("analyze.analyzed", { count: clip.analysis?.items ?? 0 }));
    else if (clip.status === "stopped") lines.push(t("analyze.stopped"));
    else if (clip.status === "failed") lines.push(t("analyze.failed"));
    else lines.push(t("analyze.not_analyzed"));
    const percent = Math.round(clip.fraction * 100);
    const unchecked = uncheckedFrames(clip);
    const failure = clip.status === "failed" && clip.failure ? t(clip.failure.code, clip.failure.params) : null;
    return {
      key: clip.id,
      signature: JSON.stringify([clip.name, clip.status, percent, lines[0], failure, unchecked]),
      build: () => h("button", {
        class: "clip-card clip-card-progress", type: "button", "aria-pressed": String(clip.status === "analyzing"),
        onClick: () => (store.getState().clips.find((c) => c.id === clip.id)?.status === "analyzed" ? openReview(clip.id) : store.dispatch({ type: "ui.select", clip_id: clip.id })),
      },
        h("span", { class: "clip-line" }, h("span", { class: "clip-name" }, nameNodes(clip.name)), clip.status === "analyzing" ? h("span", { class: "accent-text" }, percentText(t, clip.fraction)) : null),
        clip.status === "analyzing" ? h("span", { class: `bar bar-small${percent === 0 ? " is-waiting" : ""}` }, h("span", { class: "bar-fill", style: `width:${percent}%` })) : null,
        h("span", { class: "dim" }, lines[0]),
        failure ? h("span", { class: "failure" }, failure) : null,
        unchecked > 0 ? h("span", { class: "unchecked" }, t("analyze.unchecked", { count: unchecked })) : null),
    };
  }

  function renderLanes(clip) {
    // What the detectors found: an area the person added in Review is not one of their finds (it sat on the first row of the Faces lane, over a face)
    const { placed, rows } = packRows(clip.items.filter((item) => item.kind !== "manual"));
    const photo = clip.media === "photo";
    // A photo has one moment, not a timeline: its items overlap in time all the time, and a bar across the whole width for each (five faces were
    // five full-width bars) says nothing. They are drawn as small marks side by side in their lane, one row each lane, with no playhead.
    if (photo) rows.face = rows.plate = 1;
    const rowHeight = 22;
    const laneBlock = (kind, name, top) =>
      h("div", { class: "lane", style: `height:${Math.max(rows[kind], 1) * rowHeight}px;top:${top}px` }, h("span", { class: "lane-name" }, name));
    const faceHeight = Math.max(rows.face, 1) * rowHeight;
    const track = h("div", { class: "lane-track", style: `height:${faceHeight + Math.max(rows.plate, 1) * rowHeight + 12}px` });
    const marks = { face: 0, plate: 0 };
    const count = { face: 0, plate: 0 };
    for (const { lane } of placed) count[lane] += 1;
    for (const { item, lane, row: r } of placed) {
      const notSure = isNotSure(item) ? " bar-notsure" : "";
      const top = (photo ? 0 : r * rowHeight) + (lane === "plate" ? faceHeight + 12 : 0);
      if (photo) {
        // 22 px apart, closer together when there are more than the lane is wide (32 plates in one photo ran over the notes on the right)
        const n = marks[lane]++;
        const spread = `calc(${n} * (100% - 18px) / ${Math.max(count[lane] - 1, 1)})`;
        track.append(h("span", { class: `bar-item bar-${lane}${notSure} bar-mark`, style: `left:min(${n * 22}px, ${spread});top:${top}px;width:18px` }));
        continue;
      }
      const left = (item.first_frame / clip.frames) * 100;
      const width = Math.max(((item.last_frame - item.first_frame + 1) / clip.frames) * 100, 0.4);
      track.append(h("span", { class: `bar-item bar-${lane}${notSure}`, style: `left:${left}%;top:${top}px;width:${width}%` }));
    }
    if (!photo) track.append(h("span", { class: "playhead", style: `left:${(clip.frame / clip.frames) * 100}%` }));
    lanes.replaceChildren(laneBlock("face", t("analyze.lane_faces"), 0), laneBlock("plate", t("analyze.lane_plates"), faceHeight + 12), track);
  }

  function update(state) {
    const idle = state.job == null && !state.clips.some((c) => c.status === "analyzed");
    center.classList.toggle("is-idle", idle);
    heading.textContent = t("import.media_heading", { count: state.clips.length });
    reconcile(clipList, state.clips.map(clipEntry));

    const clip = state.clips.find((c) => c.status === "analyzing") ?? selectedClip(state);
    if (!clip) return;
    const live = clip.status === "analyzing";
    stage.setClip(clip);
    stage.showImage(api.previewUrl(clip.id, clip.frame));
    stage.setBoxes(clip.live);
    const zones = live && clip.stage === "second" ? clip.zones : null;
    stage.el.dataset.stage = zones ? "second" : "first";
    second.hidden = zones == null;
    if (zones) {
      secondText.textContent = t("analyze.second_look", { done: zones.done, total: zones.total });
      secondBar.firstChild.style.width = `${zones.total > 0 ? Math.round((zones.done / zones.total) * 100) : 0}%`;
    }
    title.replaceChildren(h("span", { class: "dim" }, `${t(live ? "analyze.analyzing" : "analyze.clip")} \u00b7 `), clip.name);
    title.title = clip.name;
    frameCount.textContent = frameLabel(clip, t);
    badge.hidden = !live;
    badge.textContent = [t("analyze.badge"), liveClock(clip)].filter(Boolean).join(" ");
    const counts = clip.items.length || !clip.found ? itemCounts(clip.items) : { faces: clip.found.face ?? 0, plates: clip.found.plate ?? 0, not_sure: null };  // live: the doubtful ones are not known yet
    const found = `${t("analyze.count_faces", { count: counts.faces })} \u00b7 ${t("analyze.count_plates", { count: counts.plates })}`;
    if (counts.not_sure == null) foundSummary.replaceChildren(found);
    else foundSummary.replaceChildren(`${found} \u00b7 `, h("span", { class: "notsure-text" }, t("analyze.count_notsure", { count: counts.not_sure })));
    renderLanes(clip);

    const inJob = state.clips.filter((c) => c.status !== "imported");
    const index = Math.max(1, inJob.findIndex((c) => c.id === clip.id) + 1);
    progressRows.replaceChildren(
      row(t("analyze.row_clip"), t("analyze.clip_of", { k: index, n: inJob.length })),
      row(t("analyze.row_this"), live ? etaText(clip.eta_s) : t(THIS_CLIP_WORD[clip.status] ?? "analyze.done")),
      row(t("analyze.row_all"), state.job ? etaText(remainingSeconds(state, clip.eta_s)) : allWord(inJob, t)));
    bar.firstChild.style.width = `${Math.round(overallFraction(inJob) * 100)}%`;
    // The engine reports nothing of a photo until its first pass ends (13 s for a street photo in Careful): until it does, a bar that moves says it is working.
    bar.classList.toggle("is-waiting", state.job?.kind === "analyze" && state.job.state === "running" && overallFraction(inJob) === 0 && inJob.some((c) => c.status === "analyzing"));
    const paused = state.job?.state === "paused";
    pause.textContent = t(paused ? "analyze.resume" : "analyze.pause");
    pause.disabled = state.job?.state === "stopping" || state.job == null;
    // A stop is not at once (a photo's first pass cannot be broken off: up to 8 s on a street photo in Careful): the button says it is stopping.
    stop.disabled = state.job == null || state.job.state === "stopping";
    stop.textContent = t(state.job?.state === "stopping" ? "analyze.stopping" : "analyze.stop");
    settings.textContent = settingsLine(state.options, t, state.clips.some((c) => c.media === "video"));
  }

  return { update };
}

// What a clip that is not being analyzed says on the "This clip" row: "done" is only for one that was analyzed (a stopped or a failed one said "done" too).
const THIS_CLIP_WORD = { analyzed: "analyze.done", stopped: "analyze.stopped", failed: "analyze.failed", queued: "analyze.queued", imported: "analyze.not_analyzed" };

// The "All clips" row when nothing runs: "done" only when every clip of the run was analyzed, else how many were.
function allWord(clips, t) {
  const done = clips.filter((c) => c.status === "analyzed").length;
  return done === clips.length ? t("analyze.done") : t("analyze.some_done", { done, total: clips.length });
}

// What the analysis was made with, as a line. Gap filling and the audio are about videos: a project of photos only does not mention them.
function settingsLine(o, t, hasVideo = true) {
  const detect = o.faces && o.plates ? t("settings.both") : o.faces ? t("options.faces") : t("options.plates");
  return [
    detect,
    t(`options.detection.${o.detection}`),
    hasVideo ? t("settings.gaps", { method: t(`settings.gap.${o.gap_fill}`) }) : null,
    t("settings.blur", { style: t(`options.blur.${o.blur}`), percent: Math.round(o.extra_margin * 100) }),
    hasVideo ? t("settings.audio") : null,
    t(o.remove_gps ? "settings.gps_removed" : "settings.gps_kept"),
  ].filter(Boolean).join(" \u00b7 ");
}
