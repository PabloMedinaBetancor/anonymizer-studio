// Things that move the page (a notice that comes or goes pushes the content below it) wait while a pointer button is held on the page:
// a click whose button went down on a control that then moved lands on something else, and a box being drawn would slip under the
// pointer. `after(fn)` runs `fn` at once when nothing is held, else when the button comes up (or the window loses it); the same `fn`
// asked for again while it waits runs once.
export function createPressGate(target) {
  let down = false;
  let waiting = new Set();
  function release() {
    down = false;
    const run = waiting;
    waiting = new Set();
    for (const fn of run) fn();
  }
  target.addEventListener("pointerdown", () => { down = true; }, true);
  for (const kind of ["pointerup", "pointercancel", "dragend"]) target.addEventListener(kind, release, true);
  // Only the window losing focus counts: pressing a button takes the focus from the control that had it, and that blur must not open the gate.
  target.addEventListener("blur", (event) => { if (event.target === target) release(); }, true);
  // A button released outside the window, or a lost event: any movement with no button held ends the wait.
  target.addEventListener("pointermove", (event) => { if (down && event.buttons === 0) release(); }, true);
  return {
    after(fn) {
      if (down) waiting.add(fn);
      else fn();
    },
    get held() { return down; },
  };
}
