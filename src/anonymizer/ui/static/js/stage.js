import { h } from "./dom.js";
import { drawScene } from "./scene.js";
import { createBoxLayer, createBlurLayer } from "./boxes.js";

// The frame area: a <video> when the clip plays, an <img> for a server frame or a photo, else a drawn placeholder.
export function createStage(t, { onImageError = null } = {}) {
  const canvas = h("canvas", { width: 960, height: 540, class: "stage-media", "aria-hidden": "true" });
  const video = h("video", { class: "stage-media", preload: "metadata", muted: true, playsinline: true, hidden: true });
  const image = h("img", { class: "stage-media", alt: "", hidden: true });
  const layer = createBoxLayer();
  const blur = createBlurLayer();
  // What stays on the picture when it cannot be had (the original was moved, renamed or deleted after the analysis): the broken-image icon on black said nothing,
  // and the notice that says why goes after 20 s.
  const missing = h("div", { class: "stage-missing", role: "status", hidden: true }, t("stage.missing"));
  const el = h("div", { class: "stage" }, canvas, video, image, layer.el, blur.el, missing);
  let size = { width: 1920, height: 1080 };
  drawScene(canvas);
  image.addEventListener("load", () => {
    missing.hidden = true;
  });
  image.addEventListener("error", () => {
    image.dataset.failed = "1";
    missing.hidden = false;
    onImageError?.(image.getAttribute("src"));
  });

  function show(which) {
    video.hidden = which !== "video";
    image.hidden = which !== "image";
    canvas.hidden = which !== "canvas";
    if (which !== "image") missing.hidden = true;
  }

  return {
    el,
    video,
    setClip(clip) {
      if (!clip) return;
      size = { width: clip.width, height: clip.height };
      el.style.aspectRatio = `${clip.width} / ${clip.height}`;
      el.style.setProperty("--ratio", String(clip.width / clip.height));
    },
    showVideo: () => show("video"),
    showImage(url) {
      if (!url) {
        show("canvas");
        return;
      }
      if (image.getAttribute("src") !== url) {
        missing.hidden = true;
        image.src = url;
      }
      show("image");
    },
    setBoxes: (detections) => layer.set(detections, size, t),
    setBlur: (boxes, mode) => blur.set(boxes, size, mode),
    // Mock mode and clips with no server frame: the drawn scene with the review view applied to it.
    paintScene: (options) => {
      drawScene(canvas, options);
      show("canvas");
    },
    size: () => size,
  };
}
