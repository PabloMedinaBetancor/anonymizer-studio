import { test } from "./harness.mjs";
import { strict as assert } from "assert";
import { createNoticeTimers, noticeSeconds, NOTICE_SECONDS } from "../../src/anonymizer/ui/static/js/notice-timers.js";

function fakeClock() {
  let now = 0;
  let next = 1;
  const jobs = new Map();
  return {
    setTimer: (fn, ms) => { jobs.set(next, { fn, at: now + ms }); return next++; },
    clearTimer: (id) => jobs.delete(id),
    advance(seconds) {
      now += seconds * 1000;
      for (const [id, job] of [...jobs]) if (job.at <= now) { jobs.delete(id); job.fn(); }
    },
  };
}

const notice = (id, level = "warning") => ({ id, level, code: "x", params: {} });

test("a warning goes after its time, an error stays longer", () => {
  assert.ok(NOTICE_SECONDS.error > NOTICE_SECONDS.warning);
  const clock = fakeClock();
  const gone = [];
  const timers = createNoticeTimers({ dismiss: (id) => gone.push(id), ...clock });
  timers.sync([notice(1, "warning"), notice(2, "error")]);
  clock.advance(NOTICE_SECONDS.warning);
  assert.deepEqual(gone, [1]);
  clock.advance(NOTICE_SECONDS.error - NOTICE_SECONDS.warning);
  assert.deepEqual(gone, [1, 2]);
  assert.equal(timers.pending(), 0);
});

test("a notice that is already counting is not restarted by the next render", () => {
  const clock = fakeClock();
  const gone = [];
  const timers = createNoticeTimers({ dismiss: (id) => gone.push(id), ...clock });
  timers.sync([notice(1)]);
  clock.advance(NOTICE_SECONDS.warning - 1);
  timers.sync([notice(1)]);
  clock.advance(1);
  assert.deepEqual(gone, [1]);
});

test("the pointer holds a notice, leaving starts its whole time again", () => {
  const clock = fakeClock();
  const gone = [];
  const timers = createNoticeTimers({ dismiss: (id) => gone.push(id), ...clock });
  timers.sync([notice(1)]);
  timers.hold(1);
  clock.advance(1000);
  timers.sync([notice(1)]);
  clock.advance(1000);
  assert.deepEqual(gone, []);
  timers.release(1);
  clock.advance(NOTICE_SECONDS.warning - 1);
  assert.deepEqual(gone, []);
  clock.advance(1);
  assert.deepEqual(gone, [1]);
});

test("a notice closed by hand loses its clock, and a test can shorten every time", () => {
  const clock = fakeClock();
  const timers = createNoticeTimers({ dismiss: () => {}, override: 2, ...clock });
  timers.sync([notice(1), notice(2, "error")]);
  assert.equal(timers.pending(), 2);
  timers.sync([notice(2, "error")]);
  assert.equal(timers.pending(), 1);
  assert.equal(noticeSeconds("error", 2), 2);
  assert.equal(noticeSeconds("odd"), NOTICE_SECONDS.warning);
});
