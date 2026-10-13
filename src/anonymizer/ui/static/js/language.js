import { h, svg, ICONS } from "./dom.js";
import { LANGUAGE_NAMES } from "./i18n.js";

// The language of the page when nobody has chosen one: the first of the browser's preferred languages that the Studio has (a regional
// form counts as its language: pt-BR is Portuguese, de-AT German, and any Chinese is the Simplified file, the only one there is);
// English when it has none of them. A language chosen once is the setting's, and this is not asked.
export function languageFromBrowser(preferred, supported) {
  for (const tag of preferred ?? []) {
    const lower = String(tag).toLowerCase().replace(/_/g, "-");
    const base = lower.split("-")[0];
    const found = supported.find((code) => code.toLowerCase() === lower) ?? supported.find((code) => code.toLowerCase().split("-")[0] === base);
    if (found) return found;
  }
  return "en";
}

// The language menu: a short dialog listing the languages the server has, each in its own name.
export function openLanguages({ languages, current, t, onPick }) {
  const list = h("ul", { class: "explorer-list" });
  const dialog = h("dialog", { class: "dialog language-dialog", "aria-label": t("language.title") },
    h("header", { class: "dialog-head" }, h("h2", {}, t("language.title")),
      h("button", { class: "btn-icon", type: "button", "aria-label": t("dialog.close"), onClick: () => dialog.close() }, svg(ICONS.close))),
    h("div", { class: "dialog-body" }, list));
  for (const code of languages) {
    const button = h("button", { type: "button", class: "explorer-row", "aria-pressed": String(code === current) }, LANGUAGE_NAMES[code] ?? code);
    button.addEventListener("click", () => {
      dialog.close();
      onPick(code);
    });
    list.append(h("li", {}, button));
  }
  document.body.append(dialog);
  dialog.addEventListener("close", () => dialog.remove());
  dialog.showModal();
}
