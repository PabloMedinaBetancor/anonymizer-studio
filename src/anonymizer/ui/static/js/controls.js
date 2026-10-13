import { h } from "./dom.js";

export function toggle({ label, swatch, value, onChange }) {
  const knob = h("span", { class: "toggle-track" }, h("span", { class: "toggle-knob" }));
  const button = h(
    "button",
    { class: "toggle", type: "button", role: "switch", "aria-checked": String(value) },
    h("span", { class: "toggle-label" }, swatch ? h("span", { class: `swatch swatch-${swatch}` }) : null, label),
    knob,
  );
  button.addEventListener("click", () => onChange(button.getAttribute("aria-checked") !== "true"));
  return {
    el: button,
    set(next) {
      button.setAttribute("aria-checked", String(next));
    },
  };
}

export function segmented({ options, value, label, onChange }) {
  const buttons = new Map();
  const group = h("div", { class: "segmented", role: "group", "aria-label": label });
  for (const option of options) {
    const button = h("button", { type: "button", onClick: () => onChange(option.value) }, option.label);
    buttons.set(option.value, button);
    group.append(button);
  }
  const api = {
    el: group,
    set(next) {
      for (const [key, button] of buttons) button.setAttribute("aria-pressed", String(key === next));
    },
    setLabel(key, text) {
      buttons.get(key).textContent = text;
    },
    setDisabled(key, disabled, title = "") {
      const button = buttons.get(key);
      button.disabled = disabled;
      button.title = title;
    },
  };
  api.set(value);
  return api;
}

export function select({ options, value, id, onChange }) {
  const el = h("select", { id, class: "select" }, options.map((o) => h("option", { value: o.value }, o.label)));
  el.value = value;
  el.addEventListener("change", () => onChange(el.value));
  return {
    el,
    set(next) {
      if (el.value !== next) el.value = next;
    },
  };
}

export function slider({ id, min, max, value, onChange }) {
  const el = h("input", { id, type: "range", min, max, value, class: "slider" });
  el.addEventListener("input", () => onChange(Number(el.value)));
  return {
    el,
    set(next) {
      if (Number(el.value) !== next) el.value = next;
    },
  };
}
