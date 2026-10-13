import { h } from "./dom.js";
import { clock } from "./format.js";
import { layoutTimeline, hitTimeline, frameAtX, xAtFrame, rulerTicks, needsCheck } from "./review-logic.js";

// The Review timeline: a ruler, the Faces lanes above the Plates lanes (one row per overlapping item), the playhead.
// Bars are plain spans; one pointer listener on the track does the selecting and scrubbing, so 500 items stay cheap.
const LABEL_WIDTH = 74;

export function createTimeline({ t, onSeek, onSelect }) {
  const title = h("span", { class: "section-title" });
  const legend = h("div", { class: "legend" },
    h("span", {}, h("span", { class: "swatch swatch-face" }), t("analyze.lane_faces")),
    h("span", {}, h("span", { class: "swatch swatch-plate" }), t("review.legend.plate")),
    h("span", {}, h("span", { class: "swatch swatch-manual" }), t("review.cert.manual")),
    h("span", {}, h("span", { class: "swatch swatch-notsure" }), t("review.legend.not_sure")));
  const ruler = h("div", { class: "timeline-ruler" });
  const track = h("div", { class: "timeline-track" });
  const labels = h("div", { class: "timeline-labels" });
  const playhead = h("span", { class: "playhead" });
  const marker = h("span", { class: "playhead-marker" });
  const content = h("div", { class: "timeline-content" }, labels, track);
  const scroller = h("div", { class: "timeline-scroll" }, content);
  const el = h("section", { class: "timeline", "aria-label": t("review.timeline") },
    h("div", { class: "timeline-head" }, title, legend),
    h("div", { class: "timeline-body" }, ruler, scroller));

  let data = { items: [], frames: 1, fps: 25, duration_s: 0, selectedId: null, onlyCheck: false, numbers: new Map() };
  let layout = null;
  let builtFor = null;
  let width = 0;
  let frame = 0;
  const bars = new Map();

  function measure() {
    return track.clientWidth || Math.max(300, (el.clientWidth || 1000) - LABEL_WIDTH - 12);
  }

  function itemTitle(item) {
    return `${t(`review.item.${item.kind}`, { id: data.numbers.get(item.item_id) ?? item.item_id })} · ${t("review.range", { from: clock(item.first_frame / data.fps), to: clock((item.last_frame + 1) / data.fps) })}`;
  }

  function rebuild() {
    width = measure();
    layout = layoutTimeline(data.items, data.frames, width);
    track.style.height = `${layout.height}px`;
    bars.clear();
    const nodes = layout.rects.map(({ item, lane, x, y, w, h: height }) => {
      const checkNow = needsCheck(item);
      const node = h("span", {
        class: `bar-item bar-${lane}${checkNow ? " bar-notsure" : item.blur ? "" : " bar-off"}`,
        style: `left:${x}px;top:${y}px;width:${w}px;height:${height - 4}px`,
        title: itemTitle(item),
        dataset: { id: item.item_id },
      });
      bars.set(item.item_id, node);
      return node;
    });
    labels.replaceChildren(...[
      h("span", { class: "lane-name", style: `top:${layout.lanes.face.top}px` }, t("analyze.lane_faces")),
      h("span", { class: "lane-name", style: `top:${layout.lanes.plate.top}px` }, t("analyze.lane_plates")),
      layout.lanes.manual ? h("span", { class: "lane-name", style: `top:${layout.lanes.manual.top}px` }, t("analyze.lane_added")) : null].filter(Boolean));
    track.replaceChildren(...nodes, playhead);
    ruler.replaceChildren(...rulerTicks(data.duration_s, width).map((tick) => h("span", { class: `tick${tick.x > width - 40 ? " tick-end" : ""}`, style: `left:${tick.x}px` }, clock(tick.seconds))), marker);
    markSelection();
    placePlayhead();
  }

  function markSelection() {
    for (const [id, node] of bars) node.classList.toggle("is-selected", id === data.selectedId);
  }

  function placePlayhead() {
    const x = xAtFrame(frame, width, data.frames);
    playhead.style.left = `${x}px`;
    marker.style.left = `${x}px`;
  }

  function seekFromEvent(event) {
    const rect = track.getBoundingClientRect();
    onSeek(frameAtX(event.clientX - rect.left, rect.width, data.frames));
  }

  let scrubbing = false;
  track.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const rect = track.getBoundingClientRect();
    const hit = layout ? hitTimeline(layout, event.clientX - rect.left, event.clientY - rect.top) : null;
    if (hit) onSelect(hit.item_id);
    scrubbing = true;
    track.setPointerCapture?.(event.pointerId);
    seekFromEvent(event);
  });
  track.addEventListener("pointermove", (event) => scrubbing && seekFromEvent(event));
  const stopScrub = (event) => {
    scrubbing = false;
    track.releasePointerCapture?.(event.pointerId);
  };
  track.addEventListener("pointerup", stopScrub);
  track.addEventListener("pointercancel", stopScrub);
  ruler.addEventListener("pointerdown", (event) => {
    const rect = ruler.getBoundingClientRect();
    onSeek(frameAtX(event.clientX - rect.left, rect.width, data.frames));
  });
  // Rebuilt in the next frame, not inside the observer: the rebuild changes the height of what is observed, and the browser then reports
  // "ResizeObserver loop completed with undelivered notifications" as an error of the page.
  let resizeFrame = 0;
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      if (resizeFrame) return;
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = 0;
        if (builtFor && measure() !== width) rebuild();
      });
    }).observe(scroller);
  }

  return {
    el,
    // Rebuilds only when something that changes the picture changed; a selection only moves the highlight.
    setData(next) {
      const rebuildNeeded = !builtFor || builtFor.items !== next.items || builtFor.onlyCheck !== next.onlyCheck || builtFor.frames !== next.frames;
      const selectionChanged = data.selectedId !== next.selectedId;
      data = next;
      title.textContent = t("review.timeline_items", { count: next.items.length });
      if (rebuildNeeded) {
        builtFor = { items: next.items, onlyCheck: next.onlyCheck, frames: next.frames };
        rebuild();
      } else if (selectionChanged) {
        markSelection();
      }
      if (selectionChanged || rebuildNeeded) bars.get(next.selectedId)?.scrollIntoView?.({ block: "nearest" });
    },
    setFrame(next) {
      frame = next;
      placePlayhead();
    },
  };
}
