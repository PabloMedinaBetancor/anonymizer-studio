// A drawn street scene used when no real frame is available (mock mode, screenshots of the layout).
// Detections over the drawn scene, in pixels of a 1920x1080 frame.
export const SCENE_DETECTIONS = [
  { kind: "face", state: "found", score: 0.97, box: [365, 383, 461, 491] },
  { kind: "face", state: "unsure", score: 0.58, box: [979, 383, 1075, 491] },
  { kind: "plate", state: "found", score: 0.94, box: [1315, 815, 1565, 880] },
  { kind: "plate", state: "unsure", score: 0.41, box: [134, 810, 259, 859] },
];

function rect(ctx, w, h, x, y, rw, rh, color, radius = 0) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x * w, y * h, rw * w, rh * h, radius);
  ctx.fill();
}

export function drawScene(canvas, options = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  rect(ctx, w, h, 0, 0, 1, 1, "#D6CCBF");
  for (const x of [0.03, 0.14, 0.25, 0.43, 0.54, 0.66, 0.77, 0.88]) {
    rect(ctx, w, h, x, 0.1, 0.06, 0.12, "#BDB1A2");
    if (x < 0.3 || (x > 0.5 && x < 0.6)) rect(ctx, w, h, x, 0.32, 0.06, 0.12, "#BDB1A2");
  }
  rect(ctx, w, h, 0.69, 0.25, 0.27, 0.32, "#8397A1");
  rect(ctx, w, h, 0.69, 0.25, 0.27, 0.08, "#2A3A44");
  rect(ctx, w, h, 0, 0.58, 1, 0.07, "#C9BFB0");
  rect(ctx, w, h, 0, 0.65, 1, 0.35, "#6A6D72");
  for (let i = 0; i < 8; i += 1) rect(ctx, w, h, 0.015 + i * 0.125, 0.92, 0.07, 0.02, "#E8E4DC");
  rect(ctx, w, h, 0.385, 0.2, 0.01, 0.52, "#4A5058");
  rect(ctx, w, h, 0.37, 0.2, 0.04, 0.02, "#4A5058", 2);
  for (const [x, shirt] of [[0.2, "#8A5A44"], [0.52, "#4A6F94"]]) {
    rect(ctx, w, h, x - 0.02, 0.45, 0.056, 0.2, shirt, 8);
    rect(ctx, w, h, x - 0.005, 0.36, 0.04, 0.09, "#D4A585", 20);
    rect(ctx, w, h, x - 0.005, 0.36, 0.04, 0.025, "#2A1F1A", 6);
  }
  rect(ctx, w, h, 0.04, 0.72, 0.15, 0.1, "#8A3B3B", 8);
  rect(ctx, w, h, 0.075, 0.755, 0.045, 0.03, "#F2EFEA", 2);
  rect(ctx, w, h, 0.62, 0.66, 0.35, 0.19, "#3D5A80", 10);
  rect(ctx, w, h, 0.69, 0.76, 0.12, 0.05, "#F4F1EC", 3);
  rect(ctx, w, h, 0.64, 0.7, 0.03, 0.03, "#F2E3B8", 4);
  rect(ctx, w, h, 0.93, 0.7, 0.03, 0.03, "#F2E3B8", 4);
  if (options.boxes) paintReview(ctx, canvas, options.view, options.boxes);
}

// What the Review views look like on the drawn scene: blurred patches, outlines, or half and half.
function paintReview(ctx, canvas, view, boxes) {
  const k = canvas.width / 1920;
  const pixelate = ([x0, y0, x1, y1]) => {
    const x = Math.round(x0 * k);
    const y = Math.round(y0 * k);
    const w = Math.max(2, Math.round((x1 - x0) * k));
    const h = Math.max(2, Math.round((y1 - y0) * k));
    const small = document.createElement("canvas");
    small.width = Math.max(1, Math.round(w / 9));
    small.height = Math.max(1, Math.round(h / 9));
    small.getContext("2d").drawImage(canvas, x, y, w, h, 0, 0, small.width, small.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(small, 0, 0, small.width, small.height, x, y, w, h);
  };
  if (view === "anonymized" || view === "split") {
    ctx.save();
    if (view === "split") {
      ctx.beginPath();
      ctx.rect(canvas.width / 2, 0, canvas.width / 2, canvas.height);
      ctx.clip();
    }
    for (const entry of boxes) if (entry.blur) pixelate(entry.box);
    ctx.restore();
    if (view === "split") {
      ctx.fillStyle = "#fff";
      ctx.fillRect(canvas.width / 2 - 1, 0, 2, canvas.height);
    }
  } else if (view === "boxes") {
    ctx.lineWidth = 3;
    for (const entry of boxes) {
      const [x0, y0, x1, y1] = entry.box;
      ctx.strokeStyle = entry.check ? "#f5c542" : entry.item.kind === "face" ? "#78c850" : "#f0aa3c";
      ctx.setLineDash(entry.check ? [8, 5] : []);
      ctx.strokeRect(x0 * k, y0 * k, (x1 - x0) * k, (y1 - y0) * k);
    }
    ctx.setLineDash([]);
  }
}
