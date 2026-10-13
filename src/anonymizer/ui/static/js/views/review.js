import { h, nameNodes, svg, ICONS, replaceKeepingFocus } from "../dom.js";
import { createStage } from "../stage.js";
import { createPlayer, browserCanPlay } from "../player.js";
import { createTimeline } from "../timeline.js";
import { createItemList } from "../item-list.js";
import { segmented } from "../controls.js";
import { ApiError, errorNotice } from "../api.js";
import { askConfirm } from "../confirm-dialog.js";
import { reviewClip, isReady, uncheckedFrames } from "../state.js";
import { fixed, timecode, percentText } from "../format.js";
import { SCENE_DETECTIONS } from "../scene.js";
import {
  needsCheck, doubtfulItems, bulkDecision, foundBySecondLook, visibleItems, nextToCheck, neighbourItem, clampFrame, boxFromDrag, areaFrames, certaintyKey, certaintyPercent, sceneBoxes, itemNumbers,
} from "../review-logic.js";

const VIEWS = ["original", "anonymized", "split", "boxes"];
const DURATIONS = [{ value: "2", seconds: 2 }, { value: "5", seconds: 5 }, { value: "10", seconds: 10 }, { value: "end", seconds: null }];
const BOX_CHUNK = 300;

export function mountReview(root, { store, api, t }) {
  // A picture the server could not make (the original file changed or went after the analysis) is a broken image on black: ask the server
  // why, once for each clip and analysis, and say it.
  const explained = new Set();
  async function explainPicture(url) {
    const clip = current();
    const key = `${clip?.id}:${clip?.analysis?.rev}`;
    if (!url || !clip || explained.has(key)) return;
    explained.add(key);
    try {
      const response = await fetch(url);
      if (!response.ok) store.dispatch(errorNotice(new ApiError(response.status, await response.json().catch(() => null))));
    } catch (failure) {
      explained.delete(key);  // the server is not there: the next thing the person does says so
    }
  }
  const stage = createStage(t, { onImageError: explainPicture });
  const player = createPlayer(onPlayer);
  let attachedId = null;
  let playable = false;
  let frame = 0;
  let playing = false;
  let busy = false;
  let adding = false;
  let pending = null;
  let drag = null;
  let followValue = "camera";
  let numbers = new Map();
  let filtered = { source: null, onlyCheck: null, onlySecond: null, items: [] };
  const boxCache = new Map();
  const boxRequested = new Set();
  let asking = null;  // the AbortController of the confirmation that is open, so leaving the screen answers it "no"

  // Left: clips
  const clipList = h("div", { class: "clip-list" });
  const left = h("aside", { class: "side side-left" }, h("div", { class: "side-head" }, h("span", { class: "section-title" }, t("review.clips"))), clipList);

  // Centre: toolbar, picture, controls
  const views = segmented({
    options: VIEWS.map((value) => ({ value, label: t(`review.view.${value}`) })),
    value: "anonymized",
    label: t("review.view.label"),
    onChange: (value) => store.dispatch({ type: "review.view", view: value }),
  });
  const onlyCheck = h("button", { class: "btn-small", type: "button", "aria-pressed": "false", onClick: () => store.dispatch({ type: "review.only_check", value: !store.getState().review.onlyCheck }) }, t("review.only_check"));
  const onlySecond = h("button", { class: "btn-small", type: "button", "aria-pressed": "false", onClick: () => store.dispatch({ type: "review.only_second", value: !store.getState().review.onlySecond }) }, t("review.only_second"));
  const nextCheck = h("button", { class: "btn-small", type: "button", hidden: true, dataset: { action: "next-check" }, onClick: () => goNextToCheck() }, t("review.next_check"));
  const addArea = h("button", { class: "btn-small", type: "button", "aria-pressed": "false", onClick: () => (adding ? endAdding() : beginAdding()) },
    svg(ICONS.area, 14), t("review.add_area"));
  const keepDoubtful = h("button", { class: "btn-small", type: "button", hidden: true, onClick: () => decideDoubtful(false) }, t("review.doubtful.keep_visible"));
  const blurDoubtful = h("button", { class: "btn-small", type: "button", hidden: true, onClick: () => decideDoubtful(true) }, t("review.doubtful.blur_all"));
  const toolbar = h("div", { class: "review-toolbar" }, views.el, h("span", { class: "spacer" }), keepDoubtful, blurDoubtful, nextCheck, onlyCheck, onlySecond, addArea);

  const drawLayer = h("div", { class: "draw-layer", hidden: true });
  const drawRect = h("div", { class: "draw-rect", hidden: true });
  drawLayer.append(drawRect);
  stage.el.append(drawLayer);
  const hint = h("p", { class: "draw-hint", hidden: true }, t("review.draw_hint"));

  const durationSelect = h("select", { class: "select", "aria-label": t("review.area.duration") },
    DURATIONS.map((d) => h("option", { value: d.value }, d.value === "end" ? t("review.area.to_end") : t("review.area.seconds", { seconds: d.seconds }))));
  durationSelect.value = "5";
  const follow = segmented({
    options: [{ value: "camera", label: t("review.area.follow_camera") }, { value: "fixed", label: t("review.area.follow_fixed") }],
    value: "camera",
    label: t("review.area.follow"),
    onChange: (value) => {
      followValue = value;
      follow.set(value);
    },
  });
  const durationField = h("label", { class: "field-inline" }, h("span", { class: "dim" }, t("review.area.lasts")), durationSelect);
  const areaForm = h("div", { class: "area-form", hidden: true }, durationField, follow.el,
    h("div", { class: "area-actions" },
      h("button", { class: "btn-small", type: "button", onClick: endAdding }, t("dialog.cancel")),
      h("button", { class: "btn-small btn-accent", type: "button", onClick: submitArea }, t("review.area.add"))));

  const playButton = h("button", { class: "btn-play", type: "button", "aria-label": t("player.play"), onClick: () => player.toggle() });
  const timecodeText = h("span", { class: "timecode" });
  const scrub = h("input", { class: "scrub", type: "range", min: "0", max: "0", value: "0", "aria-label": t("player.seek") });
  scrub.addEventListener("input", () => seekFrame(Number(scrub.value)));
  const controls = h("div", { class: "controls" },
    h("button", { class: "btn-icon", type: "button", "aria-label": t("player.prev_frame"), onClick: () => player.step(-1) }, svg(ICONS.prev, 14)),
    playButton,
    h("button", { class: "btn-icon", type: "button", "aria-label": t("player.next_frame"), onClick: () => player.step(1) }, svg(ICONS.next, 14)),
    timecodeText, scrub);
  const unchecked = h("p", { class: "unchecked-banner", role: "alert", hidden: true });
  const empty = h("div", { class: "empty-state" }, h("p", {}, t("review.empty")),
    h("button", { class: "btn", type: "button", onClick: () => store.dispatch({ type: "ui.tab", tab: "import" }) }, t("analyze.back_to_import")));
  const center = h("section", { class: "center" }, toolbar, unchecked,
    h("div", { class: "center-body" }, h("div", { class: "stage-wrap" }, stage.el), hint, areaForm, controls), empty);

  // Right: the selected item
  const preview = h("div", { class: "item-thumb" });
  const itemTitle = h("h2", {});
  const secondBadge = h("span", { class: "pill pill-second", hidden: true }, t("review.badge.second"));
  const itemFlags = h("div", { class: "item-flags" }, secondBadge);
  const reason = h("p", { class: "dim reason" });
  const certainty = h("span", {});
  const visible = h("span", { class: "mono-value" });
  const secondValue = h("span", {});
  const secondScore = h("div", { class: "row", hidden: true }, h("span", { class: "dim" }, t("review.second_score")), secondValue);
  const length = h("span", {});
  const blurSwitch = h("button", { class: "toggle toggle-card", type: "button", role: "switch", "aria-checked": "false", onClick: toggleBlur },
    h("span", { class: "toggle-label" }, t("review.blur_item")), h("span", { class: "toggle-track" }, h("span", { class: "toggle-knob" })));
  const goStart = h("button", { class: "btn", type: "button", onClick: () => goTo("first_frame") }, t("review.go_start"));
  const goEnd = h("button", { class: "btn", type: "button", onClick: () => goTo("last_frame") }, t("review.go_end"));
  const removeArea = h("button", { class: "btn btn-danger", type: "button", onClick: () => removeSelectedArea() }, t("review.remove_area"));
  const row = (name, value) => h("div", { class: "row" }, h("span", { class: "dim" }, name), value);
  // A photo is one moment: it has no span of time to show and no frame to go to.
  const timeRows = [row(t("review.visible"), visible), row(t("review.length"), length)];
  const goPair = h("div", { class: "button-pair" }, goStart, goEnd);
  const detail = h("div", { class: "item-detail" }, preview, itemTitle, itemFlags, reason,
    row(t("review.certainty"), certainty), secondScore, ...timeRows,
    blurSwitch, goPair, removeArea);
  const nothing = h("p", { class: "note" }, t("review.nothing_selected"));
  // A photo has no time: Space and Left/Right do nothing on it, and are not listed.
  const playKeys = h("span", {}, h("kbd", {}, t("review.key.space")), ` ${t("review.sc.play")} \u00b7 `);
  const frameKeys = h("span", {}, " \u00b7 ", h("kbd", {}, t("review.key.left")), " ", h("kbd", {}, t("review.key.right")), ` ${t("review.sc.frame")}`);
  const shortcuts = h("p", { class: "note shortcuts" }, playKeys,
    h("kbd", {}, "B"), ` ${t("review.sc.blur")} \u00b7 `,
    h("kbd", {}, "N"), ` ${t("review.sc.next")} \u00b7 `, h("kbd", {}, ","), " ", h("kbd", {}, "."), ` ${t("review.sc.item")}`, frameKeys);
  const right = h("aside", { class: "side side-right" },
    h("div", { class: "options" }, h("span", { class: "section-title" }, t("review.selected")), detail, nothing), h("div", { class: "cta" }, shortcuts));

  const timeline = createTimeline({ t, onSeek: seekFrame, onSelect: (id) => store.dispatch({ type: "review.select", item_id: id }) });
  const itemList = createItemList({ t, onSelect: (id) => store.dispatch({ type: "review.select", item_id: id }) });
  root.classList.add("page-review");
  root.append(h("div", { class: "review-top" }, left, center, right), timeline.el, itemList.el);

  // Helpers
  const current = () => reviewClip(store.getState());
  const itemsOf = (clip) => clip.items;
  const selectedItem = (state, clip) => clip?.items.find((item) => item.item_id === state.review.itemId) ?? null;
  const send = (call) => call().catch((failure) => store.dispatch(errorNotice(failure)));

  function seekFrame(next) {
    const clip = current();
    if (clip) player.seek(clampFrame(next, clip.frames) / (clip.fps || 1));
    if (clip && clip.media === "photo") onPlayer({ time: 0, playing: false });
  }

  function goTo(field) {
    const item = selectedItem(store.getState(), current());
    if (item) seekFrame(item[field]);
  }

  function toggleBlur() {
    const clip = current();
    const item = selectedItem(store.getState(), clip);
    if (!item || busy || item.manual) return;
    busy = true;
    send(() => api.setItem(clip.id, item.item_id, !item.blur).then((res) => store.dispatch({ type: "item.changed", clip_id: clip.id, item: res.item, rev: res.rev, to_check: res.to_check })))
      .finally(() => { busy = false; });
  }

  // The bulk answer for every doubtful item of the clip, after the person confirms it.
  async function decideDoubtful(blur) {
    const clip = current();
    if (!clip || busy) return;
    const ask = (kind, count) => {
      asking = new AbortController();
      return askConfirm({
        t,
        title: t(`review.doubtful.${kind}_title`),
        body: t(`review.doubtful.${kind}_body`, { count }),
        confirm: t(`review.doubtful.${kind}_confirm`),
        signal: asking.signal,
      }).finally(() => { asking = null; });
    };
    const sendDecision = (value) => {
      busy = true;
      return send(() => api.setDoubtful(clip.id, value).then((res) => store.dispatch({ type: "items.changed", clip_id: clip.id, items: res.items, rev: res.rev, to_check: res.to_check })))
        .finally(() => { busy = false; });
    };
    await bulkDecision({ blur, count: doubtfulItems(clip.items).length, ask, send: sendDecision });
  }

  // The button names what it does; the key (Delete or Backspace, which a person may press meaning something else) asks first.
  async function removeSelectedArea(askFirst = false) {
    const clip = current();
    const item = selectedItem(store.getState(), clip);
    if (!item || !item.manual || busy) return;
    if (askFirst) {
      busy = true;  // a second press while the question is open asks nothing more
      asking = new AbortController();
      const yes = await askConfirm({ t, title: t("review.remove_area_title"), body: t("review.remove_area_body"), confirm: t("review.remove_area_confirm"), signal: asking.signal }).finally(() => { asking = null; busy = false; });
      const still = selectedItem(store.getState(), current());
      if (!yes || !still || still.item_id !== item.item_id) return;
    }
    busy = true;
    send(() => api.removeArea(clip.id, item.item_id).then((res) => store.dispatch({ type: "area.removed", clip_id: clip.id, item_id: item.item_id, rev: res.rev, to_check: res.to_check })))
      .finally(() => { busy = false; });
  }

  function goNextToCheck(direction = 1) {
    const clip = current();
    if (!clip) return;
    const next = nextToCheck(itemsOf(clip), frame, direction);
    if (!next) return;
    store.dispatch({ type: "review.select", item_id: next.item_id });
    seekFrame(next.first_frame);
  }

  // "," and "." walk through every item in time order (the ones the filter shows), so the keyboard can reach one that needs no check too.
  function goNeighbour(direction) {
    const clip = current();
    if (!clip) return;
    const next = neighbourItem(filtered.items, store.getState().review.itemId, frame, direction);
    if (!next) return;
    store.dispatch({ type: "review.select", item_id: next.item_id });
    seekFrame(next.first_frame);
  }

  // Add missed area: draw a box on the picture, then say for how long it lasts
  function beginAdding() {
    const clip = current();
    if (!clip) return;
    if (playing) player.toggle();
    adding = true;
    pending = null;
    drawLayer.hidden = false;
    drawRect.hidden = true;
    renderAddArea(clip);
  }

  function endAdding() {
    adding = false;
    pending = null;
    drag = null;
    drawLayer.hidden = true;
    drawRect.hidden = true;
    const clip = current();
    if (clip) renderAddArea(clip);
  }

  function renderAddArea(clip) {
    addArea.setAttribute("aria-pressed", String(adding));
    hint.hidden = !(adding && !pending);
    areaForm.hidden = !pending;
    durationField.hidden = follow.el.hidden = clip?.media === "photo";  // a photo is one picture: it lasts no time and there is no camera to follow
  }

  function placeRect(box, size) {
    const [x0, y0, x1, y1] = box;
    drawRect.style.cssText = `left:${(x0 / size.width) * 100}%;top:${(y0 / size.height) * 100}%;width:${((x1 - x0) / size.width) * 100}%;height:${((y1 - y0) / size.height) * 100}%`;
    drawRect.hidden = false;
  }

  const point = (event) => ({ x: event.clientX, y: event.clientY });
  drawLayer.addEventListener("pointerdown", (event) => {
    if (!adding || event.button !== 0) return;
    pending = null;
    drag = point(event);
    drawLayer.setPointerCapture?.(event.pointerId);
    renderAddArea(current());
  });
  drawLayer.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const box = boxFromDrag(drag, point(event), drawLayer.getBoundingClientRect(), stage.size(), 1);
    if (box) placeRect(box, stage.size());
  });
  drawLayer.addEventListener("pointerup", (event) => {
    if (!drag) return;
    const box = boxFromDrag(drag, point(event), drawLayer.getBoundingClientRect(), stage.size());
    drag = null;
    if (!box) {
      drawRect.hidden = true;
      return;
    }
    pending = { box };
    placeRect(box, stage.size());
    renderAddArea(current());
  });

  function submitArea() {
    const clip = current();
    if (!clip || !pending || busy) return;
    const seconds = DURATIONS.find((d) => d.value === durationSelect.value).seconds;
    const frames = areaFrames(frame, seconds, clip.fps, clip.frames);
    const body = { ...frames, box: pending.box, follow: followValue };
    busy = true;
    send(() => api.addArea(clip.id, body).then((res) => {
      store.dispatch({ type: "area.added", clip_id: clip.id, item: res.item, rev: res.rev });
      store.dispatch({ type: "review.select", item_id: res.item.item_id });
      endAdding();
    })).finally(() => { busy = false; });
  }

  // The picture
  function onPlayer({ time, playing: isPlaying }) {
    const clip = current();
    if (!clip) return;
    frame = clip.media === "photo" ? 0 : clampFrame(Math.round(time * clip.fps), clip.frames);
    playing = isPlaying;
    scrub.value = String(frame);
    timecodeText.textContent = `${timecode(frame / (clip.fps || 1), clip.fps || 1)} / ${timecode(clip.duration_s, clip.fps || 1)}`;
    playButton.replaceChildren(svg(isPlaying ? ICONS.pause : ICONS.play, 16));
    playButton.setAttribute("aria-label", t(isPlaying ? "player.pause" : "player.play"));
    timeline.setFrame(frame);
    paint();
  }

  function paint() {
    const state = store.getState();
    const clip = current();
    if (!clip) return;
    const view = state.review.view;
    const rev = clip.analysis?.rev ?? 0;
    stage.setBoxes([]);
    if (playing && playable) {
      stage.showVideo();
      stage.setBlur(view === "original" ? [] : cachedBoxes(clip, frame, rev), view === "boxes" ? "outline" : view === "split" ? "split" : "blur");
      return;
    }
    stage.setBlur([], "blur");
    const url = api.previewUrl(clip.id, frame, view, 960, rev);
    if (url) {
      stage.showImage(url);
    } else {
      stage.paintScene({ view, boxes: sceneBoxes(clip.items, frame, SCENE_DETECTIONS) });
    }
  }

  function cachedBoxes(clip, atFrame, rev) {
    const chunk = Math.floor(atFrame / BOX_CHUNK) * BOX_CHUNK;
    for (const start of [chunk, chunk + BOX_CHUNK]) {
      const key = `${clip.id}:${rev}:${start}`;
      if (boxRequested.has(key) || start >= clip.frames) continue;
      if (start !== chunk && atFrame - chunk < BOX_CHUNK * 0.6) continue;
      boxRequested.add(key);
      api.boxes(clip.id, start, Math.min(start + BOX_CHUNK - 1, clip.frames - 1), rev).then((res) => boxCache.set(key, res.frames), () => boxRequested.delete(key));
    }
    return (boxCache.get(`${clip.id}:${rev}:${chunk}`)?.[String(atFrame)] ?? []).map((b) => ({ box: b.box, shape: b.shape }));
  }

  // Keyboard
  document.addEventListener("keydown", (event) => {
    if (store.getState().tab !== "review" || event.ctrlKey || event.metaKey || event.altKey) return;
    if (document.querySelector("dialog[open]")) return;  // a dialog is on top: its keys are its own (B, N or Delete must not act on the page behind it)
    if (event.target.closest?.("input, select, textarea")) return;
    const key = event.key;
    if (key === " " && !event.target.closest?.("button")) player.toggle();
    else if (key === " ") return;
    else if (key === "b" || key === "B") toggleBlur();
    else if (key === "n" || key === "N") goNextToCheck(event.shiftKey ? -1 : 1);
    else if (key === "." || key === ",") goNeighbour(key === "." ? 1 : -1);
    else if (key === "ArrowRight") player.step(event.shiftKey ? 10 : 1);
    else if (key === "ArrowLeft") player.step(event.shiftKey ? -10 : -1);
    else if (key === "Escape" && adding) endAdding();
    else if ((key === "Delete" || key === "Backspace") && !adding) removeSelectedArea(true);
    else return;
    event.preventDefault();
  });

  // Rendering from the state
  function clipCard(clip, activeId) {
    const ready = isReady(clip);
    const toCheck = clip.analysis?.to_check ?? 0;
    const analyzed = clip.status === "analyzed";
    const badge = !analyzed ? t(`review.status.${clip.status}`) : ready ? t("review.ready") : toCheck > 0 ? t("review.to_check", { count: toCheck }) : t("review.not_ready");
    return h("button", { class: "clip-card clip-card-review", type: "button", disabled: !analyzed, "aria-pressed": String(clip.id === activeId), onClick: () => store.dispatch({ type: "review.open", clip_id: clip.id }) },
      h("span", { class: "clip-name" }, nameNodes(clip.name)),
      h("span", { class: "clip-line" },
        h("span", { class: `pill ${ready ? "pill-ok" : analyzed ? "pill-warn" : ""}` }, badge),
        analyzed ? h("span", { class: "dim" }, t("review.items", { count: clip.analysis?.items ?? 0 })) : null));
  }

  function attach(clip) {
    attachedId = clip.id;
    boxCache.clear();
    boxRequested.clear();
    playable = clip.media === "video" && Boolean(clip.source_url) && browserCanPlay(clip);
    stage.setClip(clip);
    player.attach(clip, playable ? stage.video : null);
    scrub.max = String(Math.max(0, clip.frames - 1));
    controls.classList.toggle("is-photo", clip.media === "photo");
    endAdding();
  }

  function update(state) {
    const clip = reviewClip(state);
    center.classList.toggle("is-idle", clip == null);
    replaceKeepingFocus(clipList, state.clips.map((c) => clipCard(c, clip?.id)));
    if (!clip) return;
    if (attachedId !== clip.id) attach(clip);

    const { view, onlyCheck: only, onlySecond: second, itemId } = state.review;
    views.set(view);
    onlyCheck.setAttribute("aria-pressed", String(only));
    onlyCheck.disabled = !only && !clip.items.some(needsCheck);  // with nothing to check a click showed an empty list
    onlySecond.setAttribute("aria-pressed", String(second));
    onlySecond.hidden = !clip.items.some(foundBySecondLook);
    if (filtered.source !== clip.items || filtered.onlyCheck !== only || filtered.onlySecond !== second || filtered.clip !== clip.id) {
      filtered = { source: clip.items, onlyCheck: only, onlySecond: second, clip: clip.id, items: visibleItems(clip.items, only, second) };
    }
    numbers = itemNumbers(clip.items);
    const isPhoto = clip.media === "photo";
    timeline.el.hidden = isPhoto;
    controls.hidden = isPhoto;  // a photo has no time: no play button, no seek bar, no "00:00:00:00 / 00:00:00:00"
    itemList.setVisible(isPhoto);
    playKeys.hidden = frameKeys.hidden = isPhoto;
    nothing.textContent = t(isPhoto ? "review.nothing_selected_photo" : "review.nothing_selected");
    if (isPhoto) itemList.setData({ items: filtered.items, selectedId: itemId, onlyCheck: only, numbers });
    else timeline.setData({ items: filtered.items, frames: clip.frames, fps: clip.fps, duration_s: clip.duration_s, selectedId: itemId, onlyCheck: only, numbers });

    const doubtful = doubtfulItems(clip.items).length;
    keepDoubtful.hidden = blurDoubtful.hidden = doubtful === 0;
    nextCheck.hidden = !clip.items.some(needsCheck);  // N does this from the keyboard; touch has no N

    const missing = uncheckedFrames(clip);
    unchecked.hidden = missing === 0;
    unchecked.textContent = t("analyze.unchecked", { count: missing });

    const item = selectedItem(state, clip);
    detail.hidden = item == null;
    nothing.hidden = item != null;
    if (item) renderItem(item, clip);
    renderAddArea(clip);
    onPlayer({ time: player.time(), playing: player.playing() });
  }

  function renderItem(item, clip) {
    // The title says which item it is; the picture is the item itself (no second "Area #n" in a grey box when it has none).
    preview.replaceChildren(...(item.thumb_url ? [h("img", { src: item.thumb_url, alt: "" })] : []));
    preview.hidden = !item.thumb_url;
    itemTitle.textContent = t(`review.item.${item.kind}`, { id: numbers.get(item.item_id) ?? item.item_id });
    reason.textContent = item.manual ? "" : t(item.judgement.reason, item.judgement.params);
    certainty.textContent = t(certaintyKey(item), { percent: certaintyPercent(item) });
    certainty.className = needsCheck(item) ? "warn-text" : "";
    secondBadge.hidden = !foundBySecondLook(item);
    secondScore.hidden = item.second_score == null;
    secondValue.textContent = item.second_score == null ? "" : percentText(t, item.second_score);
    const fps = clip.fps || 1;
    visible.textContent = clip.media === "photo" ? "-" : t("review.range", { from: timecode(item.first_frame / fps, fps), to: timecode((item.last_frame + 1) / fps, fps) });
    const frames = item.last_frame - item.first_frame + 1;
    length.textContent = clip.media === "photo" ? t("review.photo") : t("review.length_value", { seconds: fixed(frames / fps, 1), frames });
    blurSwitch.setAttribute("aria-checked", String(item.blur));
    blurSwitch.hidden = item.manual;
    for (const el of [...timeRows, goPair]) el.hidden = clip.media === "photo";
    removeArea.hidden = !item.manual;
  }

  return { update, seek: seekFrame, leave: () => { asking?.abort(); player.stop(); endAdding(); } };
}
