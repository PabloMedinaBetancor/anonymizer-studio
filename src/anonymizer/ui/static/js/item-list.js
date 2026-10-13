import { h, replaceKeepingFocus } from "./dom.js";
import { needsCheck, certaintyKey, certaintyPercent, foundBySecondLook } from "./review-logic.js";

// Review for photos: a photo has no time axis, so instead of timeline bars the items are listed as rows
// (thumbnail, name, certainty). "Not sure" rows are striped like the timeline bars; one click selects.
export function createItemList({ t, onSelect }) {
  let numbers = new Map();
  const title = h("span", { class: "section-title" });
  const legend = h("div", { class: "legend" },
    h("span", {}, h("span", { class: "swatch swatch-face" }), t("analyze.lane_faces")),
    h("span", {}, h("span", { class: "swatch swatch-plate" }), t("review.legend.plate")),
    h("span", {}, h("span", { class: "swatch swatch-notsure" }), t("review.legend.not_sure")));
  const list = h("ul", { class: "item-list" });
  const el = h("section", { class: "timeline item-list-panel", "aria-label": t("review.items.other", { count: 0 }), hidden: true },
    h("div", { class: "timeline-head" }, title, legend), list);
  const rows = new Map();
  let builtFor = null;
  let selected = null;

  function row(item) {
    const check = needsCheck(item);
    const thumb = item.thumb_url ? h("img", { src: item.thumb_url, alt: "" }) : h("span", { class: `thumb-fallback bar-${item.kind === "face" || item.kind === "manual" ? item.kind : "plate"}` });
    const node = h("li", {},
      h("button", { type: "button", class: `item-row${check ? " is-notsure" : item.blur ? "" : " is-off"}`, dataset: { id: item.item_id, kind: item.kind }, onClick: () => onSelect(item.item_id) },
        h("span", { class: "item-row-thumb" }, thumb),
        h("span", { class: "item-row-name" }, t(`review.item.${item.kind}`, { id: numbers.get(item.item_id) ?? item.item_id }),
          foundBySecondLook(item) ? h("span", { class: "pill pill-second" }, t("review.badge.second")) : null),
        h("span", { class: `item-row-cert${check ? " warn-text" : " dim"}` }, t(certaintyKey(item), { percent: certaintyPercent(item) }))));
    rows.set(item.item_id, node.firstChild);
    return node;
  }

  function mark() {
    for (const [id, node] of rows) node.setAttribute("aria-pressed", String(id === selected));
  }

  return {
    el,
    setVisible(visible) { el.hidden = !visible; },
    setData({ items, selectedId, onlyCheck, numbers: given = numbers }) {
      numbers = given;
      title.textContent = t("review.items", { count: items.length });
      if (!builtFor || builtFor.items !== items || builtFor.onlyCheck !== onlyCheck) {
        builtFor = { items, onlyCheck };
        rows.clear();
        replaceKeepingFocus(list, items.map(row));
      }
      selected = selectedId;
      mark();
    },
  };
}
