// How long a notice stays: a warning is read in a few seconds, an error is kept longer. Pointing at a notice (or tabbing
// into it) holds it, so a long reason can be read; leaving starts its time again. The page owns the notices, this owns the clocks.
export const NOTICE_SECONDS = { warning: 10, error: 20 };
const FALLBACK_SECONDS = 10;

export function noticeSeconds(level, override = null) {
  if (override != null && override > 0) return override;
  return NOTICE_SECONDS[level] ?? FALLBACK_SECONDS;
}

export function createNoticeTimers({ dismiss, override = null, setTimer = setTimeout, clearTimer = clearTimeout }) {
  const timers = new Map();
  const held = new Set();
  const level = new Map();

  function start(id) {
    clearTimer(timers.get(id));
    timers.set(id, setTimer(() => {
      timers.delete(id);
      level.delete(id);
      held.delete(id);
      dismiss(id);
    }, noticeSeconds(level.get(id), override) * 1000));
  }

  return {
    // Called with the notices now shown: starts the clock of each new one, drops the clocks of those that went.
    sync(notices) {
      const present = new Set(notices.map((n) => n.id));
      for (const id of [...timers.keys()]) {
        if (present.has(id)) continue;
        clearTimer(timers.get(id));
        timers.delete(id);
        level.delete(id);
        held.delete(id);
      }
      for (const n of notices) {
        if (timers.has(n.id) || held.has(n.id)) continue;
        level.set(n.id, n.level);
        start(n.id);
      }
    },
    hold(id) {
      held.add(id);
      clearTimer(timers.get(id));
      timers.delete(id);
    },
    release(id) {
      if (!held.delete(id)) return;
      start(id);
    },
    pending: () => timers.size,
  };
}
