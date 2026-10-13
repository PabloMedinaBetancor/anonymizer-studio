import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { createPressGate } from "../../src/anonymizer/ui/static/js/press-gate.js";

function fakeWindow() {
  const handlers = {};
  return {
    addEventListener: (kind, fn) => { (handlers[kind] = handlers[kind] || []).push(fn); },
    fire: (kind, event = {}) => (handlers[kind] ?? []).forEach((fn) => fn(event)),
  };
}

test("with no button held a change runs at once", () => {
  const gate = createPressGate(fakeWindow());
  let ran = 0;
  gate.after(() => { ran += 1; });
  assert.equal(ran, 1);
});

test("a change waits for the button to come up, and runs once however often it is asked", () => {
  const target = fakeWindow();
  const gate = createPressGate(target);
  let ran = 0;
  const change = () => { ran += 1; };
  target.fire("pointerdown");
  gate.after(change);
  gate.after(change);
  assert.equal(ran, 0);
  assert.equal(gate.held, true);
  target.fire("pointerup");
  assert.equal(ran, 1);
  assert.equal(gate.held, false);
  gate.after(change);
  assert.equal(ran, 2);
});

test("a button released outside the window, or a lost window, does not leave a change waiting for ever", () => {
  const target = fakeWindow();
  const gate = createPressGate(target);
  let ran = 0;
  target.fire("pointerdown");
  gate.after(() => { ran += 1; });
  target.fire("pointermove", { buttons: 1 });
  assert.equal(ran, 0);
  target.fire("pointermove", { buttons: 0 });
  assert.equal(ran, 1);
  target.fire("pointerdown");
  gate.after(() => { ran += 1; });
  target.fire("blur", { target });
  assert.equal(ran, 2);
});

test("the focus leaving a control when a button is pressed does not open the gate", () => {
  const target = fakeWindow();
  const gate = createPressGate(target);
  let ran = 0;
  target.fire("pointerdown");
  target.fire("blur", { target: { tagName: "INPUT" } });  // the control that had the focus, as the press takes it
  gate.after(() => { ran += 1; });
  assert.equal(ran, 0);
  assert.equal(gate.held, true);
  target.fire("pointerup");
  assert.equal(ran, 1);
});
