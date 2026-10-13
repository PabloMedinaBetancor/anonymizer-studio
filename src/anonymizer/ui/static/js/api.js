// The only way the page talks to the engine: HTTP calls and a server-sent event stream on the local server.
// Shapes are those of docs/UI-PROTOCOL.md.
export class ApiError extends Error {
  constructor(status, body) {
    super(`HTTP ${status} ${body?.code ?? ""}`.trim());
    this.status = status;
    this.code = body?.code ?? "error.internal";
    this.params = body?.params ?? {};
  }
}

// Which window of the Studio a request comes from: the server tells every window about a change of settings and says whose it was,
// so that the one that made it does not take its own change back as news.
export const CLIENT_ID = (globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)).replace(/[^A-Za-z0-9-]/g, "").slice(0, 40);

// Whether the Studio answers at all. A request that never got an answer (it failed before or while it left) asks the Studio once whether it is
// there: if it is not ("gone": it stopped, and the browser's own text would be English), the page says so in its own language and stops offering
// what cannot work (main.js). One the Studio refuses as not allowed (403: an address without the session, a bookmark of an earlier start) is
// "forbidden": the Studio is there and this page cannot use it. A request that failed for another reason (a file the browser could not read)
// says nothing about it.
const connection = { status: "ok", checking: false, listeners: { lost: null, found: null } };

function setStatus(status) {
  if (connection.status === status) return;
  connection.status = status;
  if (status === "ok") connection.listeners.found?.();
  else connection.listeners.lost?.(status);
}

async function checkConnection() {
  if (connection.checking) return;
  connection.checking = true;
  try {
    const answer = await fetch("api/state");
    setStatus(answer.status === 403 ? "forbidden" : "ok");
  } catch {
    setStatus("gone");
  } finally {
    connection.checking = false;
  }
}

async function request(method, url, { json, body, headers } = {}) {
  const init = { method, headers: { ...headers } };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  let response;
  try {
    response = await fetch(url, init);
  } catch (failure) {
    checkConnection();
    throw failure;
  }
  const text = await response.text();
  const parsed = text ? JSON.parse(text) : null;
  if (!response.ok) {
    if (response.status === 403 && parsed?.code === "error.request.forbidden") setStatus("forbidden");
    throw new ApiError(response.status, parsed);
  }
  return parsed;
}

export function createHttpApi() {
  return {
    state: () => request("GET", "api/state"),
    settings: () => request("GET", "api/settings"),
    putSettings: (change) => request("PUT", "api/settings", { json: change, headers: { "X-Studio-Client": CLIENT_ID } }),
    browse: (path) => request("GET", `api/browse${path == null ? "" : `?path=${encodeURIComponent(path)}`}`),
    importPaths: (paths) => request("POST", "api/clips", { json: { paths } }),
    upload: (file) =>
      request("PUT", `api/upload?name=${encodeURIComponent(file.name)}`, { body: file, headers: { "Content-Type": "application/octet-stream" } }),
    analysis: (clipId) => request("GET", `api/clips/${clipId}/analysis`),
    analyze: (clips, options, discard = false) => request("POST", "api/job/analyze", { json: { clips, options, ...(discard ? { discard: true } : {}) } }),
    renderPlan: (clips, options) => request("POST", "api/render/plan", { json: { clips, options } }),
    render: (clips, options) => request("POST", "api/render", { json: { clips, options } }),
    samples: () => request("GET", "api/samples"),
    importSamples: (samples) => request("POST", "api/clips", { json: { samples } }),
    pause: () => request("POST", "api/job/pause"),
    resume: () => request("POST", "api/job/resume"),
    stop: () => request("POST", "api/job/stop"),
    previewUrl: (clipId, frame, view = "original", width = 960, rev = null) =>
      `api/clips/${clipId}/preview?frame=${frame}&view=${view}&w=${width}${rev == null ? "" : `&rev=${rev}`}`,
    boxes: (clipId, from, to, rev) => request("GET", `api/clips/${clipId}/boxes?from=${from}&to=${to}&rev=${rev}`),
    setItem: (clipId, itemId, blur) => request("PATCH", `api/clips/${clipId}/items/${itemId}`, { json: { blur } }),
    setDoubtful: (clipId, blur) => request("POST", `api/clips/${clipId}/doubtful`, { json: { blur } }),
    addArea: (clipId, area) => request("POST", `api/clips/${clipId}/areas`, { json: area }),
    removeClip: (clipId) => request("DELETE", `api/clips/${clipId}`),
    removeArea: (clipId, itemId) => request("DELETE", `api/clips/${clipId}/areas/${itemId}`),
    // `onLost("gone" | "forbidden")` is called once each time the Studio stops answering (it stopped by itself, or the computer slept past its idle wait) and
    // `onFound` when it answers again: the stream fails, and a request that fails too means it is not there; a stream that only dropped on a
    // live Studio says nothing.
    onEvent(handler, onLost, onFound) {
      connection.listeners = { lost: onLost, found: onFound };
      if (connection.status !== "ok") onLost?.(connection.status);
      const source = new EventSource("api/events");
      source.onmessage = (event) => handler(JSON.parse(event.data));
      source.onerror = () => checkConnection();
      source.onopen = () => setStatus("ok");
      return () => source.close();
    },
  };
}

export function errorNotice(failure) {
  if (failure instanceof ApiError) return { type: "job.error", code: failure.code, params: failure.params };
  // The browser's own reason ("Failed to fetch", "NetworkError when attempting to fetch resource") is English whatever the page's language: not shown.
  return { type: "job.error", code: "error.network", params: {} };
}
