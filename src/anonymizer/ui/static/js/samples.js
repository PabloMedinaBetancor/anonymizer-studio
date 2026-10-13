import { h, svg, ICONS } from "./dom.js";
import { clock } from "./format.js";

// The public clips of the demo (docs/UI-PROTOCOL.md section 4, "samples"): each shows its licence and author.
export function openSamples({ api, t, onPick }) {
  const chosen = new Set();
  const list = h("ul", { class: "explorer-list sample-list" });
  const error = h("p", { class: "error", role: "alert", hidden: true });
  const pickButton = h("button", { class: "btn-primary", type: "button", disabled: true }, t("samples.add"));
  const dialog = h("dialog", { class: "dialog explorer", "aria-label": t("samples.title") },
    h("header", { class: "dialog-head" }, h("h2", {}, t("samples.title")),
      h("button", { class: "btn-icon", type: "button", "aria-label": t("dialog.close"), onClick: () => dialog.close() }, svg(ICONS.close))),
    h("div", { class: "dialog-body" }, h("p", { class: "dim" }, t("samples.intro")), error, list),
    h("footer", { class: "dialog-foot" },
      h("button", { class: "btn", type: "button", onClick: () => dialog.close() }, t("dialog.cancel")), pickButton));
  document.body.append(dialog);
  dialog.addEventListener("close", () => dialog.remove());

  function row(sample) {
    const button = h("button", { type: "button", class: "explorer-row", "aria-pressed": "false" },
      svg(sample.media === "photo" ? ICONS.image : ICONS.film, 16),
      h("span", { class: "sample-text" }, h("span", {}, sample.name),
        h("span", { class: "dim mono" }, [sample.media === "photo" ? t("import.photo") : clock(sample.duration_s), sample.licence, sample.author].filter(Boolean).join(" · "))));
    button.addEventListener("click", () => {
      if (chosen.has(sample.sample_id)) chosen.delete(sample.sample_id);
      else chosen.add(sample.sample_id);
      button.setAttribute("aria-pressed", String(chosen.has(sample.sample_id)));
      pickButton.disabled = chosen.size === 0;
    });
    return h("li", {}, button);
  }

  pickButton.addEventListener("click", () => {
    const ids = [...chosen];
    dialog.close();
    onPick(ids);
  });
  dialog.showModal();
  // The list comes from the server: until it has, the dialog says so (it was an empty box), and says it when there is none to offer.
  const loading = h("li", { class: "dim sample-note", role: "status" }, t("samples.loading"));
  list.replaceChildren(loading);
  api.samples().then(
    (samples) => list.replaceChildren(...(samples.length > 0 ? samples.map(row) : [h("li", { class: "dim sample-note", role: "status" }, t("samples.empty"))])),
    (failure) => {
      list.replaceChildren();
      error.textContent = t(failure.code ?? "error.internal", failure.params);
      error.hidden = false;
    },
  );
}
