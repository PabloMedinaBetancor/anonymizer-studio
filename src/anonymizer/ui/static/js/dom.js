export function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2).toLowerCase(), value);
    else if (name === "dataset") Object.assign(node.dataset, value);
    else if (name === "style") node.style.cssText = value;
    else node.setAttribute(name, value === true ? "" : value);
  }
  node.append(...children.flat().filter((child) => child != null && child !== false));
  return node;
}

// The pieces of a file name with a place to break a line after each _ - . or space: a long name wraps there and not in the middle of a word
// (the text is the name; a <wbr> adds nothing to it, so copying the name copies the name).
export function nameNodes(name) {
  const nodes = [];
  let piece = "";
  for (const character of name) {
    piece += character;
    if ("_-. ".includes(character)) {
      nodes.push(piece, document.createElement("wbr"));
      piece = "";
    }
  }
  if (piece) nodes.push(piece);
  return nodes;
}

// Brings `parent` to show `entries` ({key, signature, build}) in order and leaves alone the node of every entry whose signature has not changed.
// A list that is built again whole on every progress event takes a click with it when the pointer went down on a node that is gone by the time
// it comes up (the click lands on the list, not on the button): about one click in ten during an analysis.
export function reconcile(parent, entries) {
  const old = new Map();
  for (const node of parent.children) if (node.__key !== undefined) old.set(node.__key, node);
  const nodes = entries.map(({ key, signature, build }) => {
    const kept = old.get(key);
    if (kept && kept.__signature === signature) return kept;
    const made = build();
    made.__key = key;
    made.__signature = signature;
    return made;
  });
  const same = nodes.length === parent.children.length && nodes.every((node, i) => parent.children[i] === node);
  if (!same) parent.replaceChildren(...nodes);
}

const FOCUSABLE = "button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";

// Puts `nodes` in `parent` in place of what is there. When the keyboard focus was on a control inside `parent`, it goes to the control that
// takes the same place in the new list: a node that is replaced takes the focus with it, so after Space on a checkbox the next Tab started at the
// top of the list again, and a second Space acted on nothing.
export function replaceKeepingFocus(parent, nodes) {
  const active = document.activeElement;
  const at = active && active !== document.body && parent.contains(active) ? [...parent.querySelectorAll(FOCUSABLE)].indexOf(active) : -1;
  parent.replaceChildren(...nodes);
  if (at < 0) return;
  const now = [...parent.querySelectorAll(FOCUSABLE)];
  now[Math.min(at, now.length - 1)]?.focus({ preventScroll: true });
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function svg(markup, size = 14) {
  const wrapper = document.createElementNS(SVG_NS, "svg");
  wrapper.setAttribute("viewBox", "0 0 24 24");
  wrapper.setAttribute("width", size);
  wrapper.setAttribute("height", size);
  wrapper.setAttribute("aria-hidden", "true");
  wrapper.innerHTML = markup;
  return wrapper;
}

export const ICONS = {
  plus: '<g stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></g>',
  prev: '<g fill="currentColor"><polygon points="19 20 9 12 19 4 19 20"/><rect x="4" y="5" width="2.5" height="14"/></g>',
  next: '<g fill="currentColor"><polygon points="5 4 15 12 5 20 5 4"/><rect x="17.5" y="5" width="2.5" height="14"/></g>',
  play: '<polygon fill="currentColor" points="7,4 20,12 7,20"/>',
  pause: '<g fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></g>',
  folder: '<path fill="currentColor" d="M3 6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  area: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 9V5h4M16 5h4v4M20 15v4h-4M8 19H4v-4"/><line x1="12" y1="9" x2="12" y2="15"/><line x1="9" y1="12" x2="15" y2="12"/></g>',
  film: '<g fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="8" y1="4" x2="8" y2="20"/><line x1="16" y1="4" x2="16" y2="20"/></g>',
  image: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6" fill="currentColor" stroke="none"/><path d="M4 18l5-5 4 4 3-3 4 4"/></g>',
  blocked: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8"/><line x1="6.5" y1="6.5" x2="17.5" y2="17.5"/></g>',
  close: '<g stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></g>',
  sliders:
    '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></g>',
};
