import { h, svg, ICONS } from "./dom.js";
import { errorNotice } from "./api.js";

function join(dir, name) {
  const sep = dir.includes("\\") && !dir.includes("/") ? "\\" : "/";
  return dir.endsWith(sep) ? dir + name : dir + sep + name;
}

// Folder browser backed by the server (a browser tab cannot see real paths).
export function openExplorer({ api, t, onPick }) {
  const chosen = new Set();
  const path = h("div", { class: "explorer-path mono" });
  const list = h("ul", { class: "explorer-list" });
  const error = h("p", { class: "error", role: "alert", hidden: true });
  const pickButton = h("button", { class: "btn-primary", type: "button" }, t("explorer.import"));
  const dialog = h("dialog", { class: "dialog explorer", "aria-label": t("explorer.title") },
    h("header", { class: "dialog-head" }, h("h2", {}, t("explorer.title")),
      h("button", { class: "btn-icon", type: "button", "aria-label": t("dialog.close"), onClick: () => dialog.close() }, svg(ICONS.close))),
    h("div", { class: "dialog-body" }, path, error, list),
    h("footer", { class: "dialog-foot" },
      h("button", { class: "btn", type: "button", onClick: () => dialog.close() }, t("dialog.cancel")), pickButton));
  document.body.append(dialog);
  dialog.addEventListener("close", () => dialog.remove());

  let here = "";

  // "Import" with nothing ticked imports the folder on show. On the list of places there is no folder: the server would take its own
  // working directory for the empty path, so the button waits until a folder is open or a file is ticked.
  function syncPick() {
    pickButton.disabled = chosen.size === 0 && here === "";
  }

  async function go(target) {
    error.hidden = true;
    let listing;
    try {
      listing = await api.browse(target);
    } catch (failure) {
      // A server that is not there at all (stopped, or the connection lost) is not an internal error of the server.
      const notice = errorNotice(failure);
      error.textContent = t("explorer.failed", { reason: t(notice.code, notice.params) });
      error.hidden = false;
      return;
    }
    here = listing.path ?? "";
    chosen.clear();
    syncPick();
    path.textContent = here;
    const roots = target == null ? (listing.roots ?? []).map((r) => ({ name: r.name === "Home" ? t("explorer.home") : r.name, kind: "dir", path: r.path })) : [];
    const entries = listing.entries.map((e) => ({ ...e, path: join(listing.path, e.name) }));
    // replaceChildren writes a null as the text "null": leave out what is not there
    list.replaceChildren(...[
      listing.parent != null ? row({ name: "..", kind: "dir", path: listing.parent }) : null,
      ...[...roots, ...entries].map(row),
      listing.truncated ? h("li", { class: "dim" }, t("explorer.truncated")) : null,
    ].filter(Boolean));
  }

  function row(item) {
    const isDir = item.kind === "dir";
    const unsupported = item.kind === "unsupported";
    const icon = isDir ? ICONS.folder : unsupported ? ICONS.blocked : item.kind === "photo" ? ICONS.image : ICONS.film;
    const button = h("button", { type: "button", class: "explorer-row", dataset: { kind: item.kind }, "aria-pressed": "false", disabled: unsupported, title: unsupported ? t(item.reason ?? "error.media.unsupported", { name: item.name, extension: item.name.slice(item.name.lastIndexOf(".")) }) : null },
      svg(icon, 16), h("span", {}, item.name));
    button.addEventListener("click", (event) => {
      // The second click of a double click (the way a folder is opened in any file dialog) falls on whatever the new list has under the pointer:
      // it is not a second choice.
      if (event.detail > 1) return;
      if (isDir) {
        go(item.path);
        return;
      }
      if (chosen.has(item.path)) chosen.delete(item.path);
      else chosen.add(item.path);
      button.setAttribute("aria-pressed", String(chosen.has(item.path)));
      syncPick();
    });
    return h("li", {}, button);
  }

  pickButton.addEventListener("click", () => {
    const paths = chosen.size > 0 ? [...chosen] : [here];
    dialog.close();
    onPick(paths);
  });
  dialog.showModal();
  syncPick();
  go(null);
}
