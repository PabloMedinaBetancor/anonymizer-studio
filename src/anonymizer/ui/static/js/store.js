export function createStore(initial, reduce) {
  let state = initial;
  const listeners = new Set();
  return {
    getState: () => state,
    dispatch(message) {
      const next = reduce(state, message);
      if (next === state) return;
      state = next;
      for (const listener of listeners) listener(state, message);
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
