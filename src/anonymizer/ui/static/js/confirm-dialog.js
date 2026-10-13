import { h } from "./dom.js";

// Own dialog instead of window.confirm: resolves true only when the person presses the confirm button. Escape, a click outside,
// Cancel and the close button all answer "no". The browser closes a modal dialog on Escape only after a click or key press
// of the person, so the key is handled here as well (a driven or an unfocused page would otherwise keep the dialog open).
// `signal` closes it from outside as a "no", which is what leaving the screen does.
export function askConfirm({ t, title, body, confirm, signal = null }) {
  return new Promise((resolve) => {
    let answer = false;
    const finish = (value) => {
      answer = value;
      if (dialog.open) dialog.close();
      else dialog.dispatchEvent(new Event("close"));
    };
    const dialog = h("dialog", { class: "dialog confirm-dialog", "aria-label": title },
      h("header", { class: "dialog-head" }, h("h2", {}, title)),
      h("div", { class: "dialog-body" }, h("p", {}, body)),
      h("footer", { class: "dialog-foot" },
        h("button", { class: "btn", type: "button", onClick: () => finish(false) }, t("dialog.cancel")),
        h("button", { class: "btn-primary", type: "button", onClick: () => finish(true) }, confirm)));
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      document.removeEventListener("keydown", onKey, true);
      dialog.remove();
      resolve(answer);
    };
    function onKey(event) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      finish(false);
    }
    dialog.addEventListener("close", done);
    dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      finish(false);
    });
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) finish(false);  // the dimmed area around the box
    });
    document.addEventListener("keydown", onKey, true);
    signal?.addEventListener("abort", () => finish(false), { once: true });
    document.body.append(dialog);
    dialog.showModal();
    dialog.querySelector(".btn").focus();
  });
}
