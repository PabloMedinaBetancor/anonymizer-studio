import { h } from "./dom.js";

// Live detections over a frame. Boxes arrive as [x0, y0, x1, y1] in source pixels; an unsure one is dashed.
export function createBoxLayer() {
  const el = h("div", { class: "box-layer" });
  return {
    el,
    set(detections, size, t) {
      el.replaceChildren(
        ...detections.map((d) => {
          const [x0, y0, x1, y1] = d.box;
          const unsure = d.state === "unsure";
          const label = `${t(`boxes.${d.kind}`)}${unsure ? "?" : ""} ${Math.round(d.score * 100)}`;
          return h(
            "div",
            {
              class: `box box-${d.kind}${unsure ? " box-notsure" : ""}`,
              style: `left:${(x0 / size.width) * 100}%;top:${(y0 / size.height) * 100}%;width:${((x1 - x0) / size.width) * 100}%;height:${((y1 - y0) / size.height) * 100}%`,
            },
            h("span", { class: "box-label" }, label),
          );
        }),
      );
    },
  };
}

// The blur over a playing <video>: one div per box that blurs what is behind it, or outlines it in the Boxes view.
// Exact blur comes from the server's preview when paused; this is the approximation of protocol section 5.
export function createBlurLayer() {
  const el = h("div", { class: "blur-layer" });
  return {
    el,
    set(boxes, size, mode) {
      el.dataset.mode = mode;
      el.replaceChildren(
        ...boxes.map(({ box, shape }) => {
          const [x0, y0, x1, y1] = box;
          return h("div", {
            class: `blur-box${shape === "ellipse" ? " blur-ellipse" : ""}`,
            style: `left:${(x0 / size.width) * 100}%;top:${(y0 / size.height) * 100}%;width:${((x1 - x0) / size.width) * 100}%;height:${((y1 - y0) / size.height) * 100}%`,
          });
        }),
      );
    },
  };
}
