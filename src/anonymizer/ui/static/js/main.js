import { createStore } from "./store.js";
import { initialState, reduce, projectName, analyzedClips } from "./state.js";
import { loadI18n, LANGUAGE_NAMES } from "./i18n.js";
import { openLanguages, languageFromBrowser } from "./language.js";
import { createHttpApi, errorNotice, CLIENT_ID } from "./api.js";
import { createMockApi } from "./mock.js";
import { h, svg, ICONS, reconcile } from "./dom.js";
import { mountImport } from "./views/import.js";
import { mountAnalyze } from "./views/analyze.js";
import { mountReview } from "./views/review.js";
import { mountExport } from "./views/export.js";
import { setNumberLocale } from "./format.js";
import { createNoticeTimers } from "./notice-timers.js";
import { createPressGate } from "./press-gate.js";

const params = new URLSearchParams(location.search);

async function start() {
  const mock = params.get("mock");
  const api = mock ? createMockApi({ ...(mock === "fast" ? { tick: 40, step: 0.05 } : {}), demo: params.get("demo") === "1" }) : createHttpApi();
  // The page language: ?lang= wins (also "pseudo"), then the language the person chose (the setting), then the browser's own when the Studio
  // has it, then English. A setting of "en" is a choice; one of null is none made yet; a language the Studio does not have counts as none.
  // A Studio that does not answer (stopped, or refusing this page) cannot list its languages: the files of all seven are public, so the page is still in the person's.
  const supported = (await api.state().catch(() => null))?.languages ?? Object.keys(LANGUAGE_NAMES).filter((code) => code !== "pseudo");
  const known = (code) => (typeof code === "string" && (supported.includes(code) || code === "pseudo") ? code : null);
  const chosen = known(params.get("lang")) ?? known((await api.settings().catch(() => null))?.language);
  const saved = chosen ?? languageFromBrowser(navigator.languages?.length ? navigator.languages : [navigator.language], supported);
  const i18n = await loadI18n(saved);
  const t = i18n.t;
  setNumberLocale(i18n.locale);
  document.documentElement.lang = i18n.locale === "en" && saved === "pseudo" ? "en" : i18n.locale;
  const store = createStore(initialState(), reduce);
  const app = document.getElementById("app");

  const tabs = [
    { id: "import", enabled: () => true },
    { id: "analyze", enabled: (s) => s.job != null || s.clips.some((c) => c.status === "analyzed") },
    { id: "review", enabled: (s) => analyzedClips(s).length > 0 },
    { id: "export", enabled: (s) => analyzedClips(s).length > 0 },
  ];
  const nav = h("nav", { class: "tabs", "aria-label": t("nav.label") });
  const tabButtons = tabs.map((tab) => {
    const button = h("button", { type: "button", onClick: () => store.dispatch({ type: "ui.tab", tab: tab.id }) }, t(`nav.${tab.id}`));
    nav.append(button);
    return button;
  });
  const languageButton = h("button", { class: "btn-lang", type: "button", "aria-label": t("language.title") }, saved === "pseudo" ? "PS" : saved.split("-")[0].toUpperCase());
  languageButton.addEventListener("click", () => {
    const languages = store.getState().info?.languages ?? ["en"];
    openLanguages({
      languages: params.get("lang") === "pseudo" ? [...languages, "pseudo"] : languages, current: saved, t,
      onPick: async (code) => {
        await api.putSettings({ language: code }).catch(() => null);
        const next = new URLSearchParams(location.search);
        next.set("lang", code);
        location.search = next.toString();
      },
    });
  });
  const projectLabel = h("span", { class: "project" });
  // In the page flow, under the top bar: a notice never lies over a control, and it goes by itself (notice-timers.js).
  const notices = h("div", { class: "notices", role: "status" });
  const noticeSeconds = Number(params.get("notice_seconds")) || null;  // a test shortens the clocks
  const header = h("header", { class: "topbar" },
    h("div", { class: "brand" }, h("span", { class: "logo" }, h("span", { class: "logo-dot" })), h("strong", {}, t("app.name")), h("span", { class: "dim" }, "/"), projectLabel),
    nav,
    h("div", { class: "top-right" }, languageButton,
      h("button", { class: "btn-icon bordered", type: "button", "aria-label": t("app.preferences"), title: t("app.preferences"), onClick: openPreferences }, svg(ICONS.sliders, 15))));
  const pages = { import: h("main", { class: "page" }), analyze: h("main", { class: "page", hidden: true }), review: h("main", { class: "page", hidden: true }), export: h("main", { class: "page page-export", hidden: true }) };
  // The Studio is not there: a bar that stays (a notice goes by itself) says so and how to start it again, and the screens under it are inert.
  const lostTitle = h("strong", {});
  const lostHow = h("span", {});
  const lostBar = h("div", { class: "lost-bar", role: "alert", hidden: true }, lostTitle, lostHow);
  app.append(header, lostBar, notices, ...Object.values(pages));

  // The preferences are the options on the Import screen (what to detect, how carefully, the blur, the output): the button goes there.
  function openPreferences() {
    store.dispatch({ type: "ui.tab", tab: "import" });
    pages.import.querySelector(".side-right .options")?.scrollIntoView?.({ block: "start" });
  }

  // A file dropped where nothing takes it (another screen, the top bar) must not make the browser open it and leave the Studio. Only the
  // Import screen takes a drop (views/import.js); on the others the pointer says so.
  window.addEventListener("dragover", (event) => {
    if (![...(event.dataTransfer?.types ?? [])].includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "none";
  });
  window.addEventListener("drop", (event) => {
    if ([...(event.dataTransfer?.types ?? [])].includes("Files")) event.preventDefault();
  });

  const ctx = { store, api, t, openReview: (clipId) => store.dispatch({ type: "review.open", clip_id: clipId }) };
  const views = {
    import: mountImport(pages.import, ctx),
    analyze: mountAnalyze(pages.analyze, ctx),
    review: mountReview(pages.review, ctx),
    export: mountExport(pages.export, ctx),
  };

  // The screen of the control that had the keyboard focus last (null after a pointer press: the person is not on the keyboard).
  let lastFocusedScreen = null;
  document.addEventListener("focusin", (event) => { lastFocusedScreen = Object.keys(pages).find((id) => pages[id].contains(event.target)) ?? null; });
  document.addEventListener("pointerdown", () => { lastFocusedScreen = null; }, true);
  const pressGate = createPressGate(window);
  const noticeTimers = createNoticeTimers({ override: noticeSeconds, dismiss: (id) => store.dispatch({ type: "ui.notice.dismiss", id }) });
  let shownTab = null;
  function render(state) {
    const leaving = shownTab && shownTab !== state.tab ? shownTab : null;
    if (leaving) views[leaving].leave?.();
    shownTab = state.tab;
    projectLabel.textContent = projectName(state, t("app.untitled"));
    tabs.forEach((tab, i) => {
      const button = tabButtons[i];
      button.disabled = !tab.enabled(state);
      if (tab.id === state.tab) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    for (const [id, page] of Object.entries(pages)) page.hidden = id !== state.tab;
    // The control that was pressed on the screen just left (Analyze, a clip card, Render) is hidden now: the focus goes to the new screen's tab, not to nowhere.
    // A button that goes dead when pressed (Analyze, Render) drops the focus to the page body before the screen changes: the last control that had it says where it was.
    if (leaving && (pages[leaving].contains(document.activeElement) || (document.activeElement === document.body && lastFocusedScreen === leaving))) {
      tabButtons[tabs.findIndex((tab) => tab.id === state.tab)]?.focus({ preventScroll: true });
    }
    lastFocusedScreen = null;
    views[state.tab].update(state);
    lostBar.hidden = !state.lost;
    if (state.lost) {
      lostTitle.textContent = t(state.lost === "forbidden" ? "lost.title_forbidden" : "lost.title");
      lostHow.textContent = t(state.lost === "forbidden" ? "error.request.forbidden" : state.info?.mode === "demo" ? "lost.how_demo" : "lost.how");
    }
    app.classList.toggle("is-lost", state.lost);
    for (const part of [header, ...Object.values(pages)]) part.inert = state.lost;
    // The list of notices changes the height of the bar above the page: not while a button is held down on the page.
    pressGate.after(flushNotices);
    noticeTimers.sync(state.notices);
  }

  function flushNotices() {
    showNotices(store.getState());
  }

  function showNotices(state) {
    reconcile(notices, state.notices.map((n) => ({
      key: n.id,
      signature: `${n.level}|${n.code}|${JSON.stringify(n.params)}`,
      build: () =>
        h("div", { class: `notice notice-${n.level}`, dataset: { id: n.id }, onMouseenter: () => noticeTimers.hold(n.id), onMouseleave: () => noticeTimers.release(n.id), onFocusin: () => noticeTimers.hold(n.id), onFocusout: () => noticeTimers.release(n.id) }, t(n.code, n.params),
          h("button", { class: "btn-icon", type: "button", "aria-label": t("dialog.close"), onClick: () => store.dispatch({ type: "ui.notice.dismiss", id: n.id }) }, svg(ICONS.close, 12))),
    })));
  }

  // Declared before the first subscription: `api.onEvent` below tells a page that opens with the Studio unreachable at once, and that dispatch
  // reaches `syncSettings` while these are still not initialised ("can't access lexical declaration 'sentSettings' before initialization").
  let settingsTimer = 0;
  let sentSettings = null;
  let pendingSettings = {};
  store.subscribe(render);
  store.subscribe(syncSettings);
  store.subscribe(loadMissingItems);
  api.onEvent(
    (message) => (message.type === "state.resync" ? loadState() : message.type === "settings.changed" ? settingsChangedElsewhere(message) : store.dispatch(message)),
    (status) => store.dispatch({ type: "server.lost", status }),
    () => {
      store.dispatch({ type: "server.found" });
      loadState();
    },
  );

  // Only what this window changed is sent: sending every option would put back, in the server, what another window of the Studio changed
  // since this one loaded (a blur style, a margin).
  function syncSettings(state) {
    if (sentSettings === null || state.options === sentSettings) return;
    for (const key of Object.keys(state.options)) if (state.options[key] !== sentSettings[key]) pendingSettings[key] = state.options[key];
    sentSettings = state.options;
    if (Object.keys(pendingSettings).length === 0) return;
    clearTimeout(settingsTimer);
    settingsTimer = setTimeout(() => {
      const change = pendingSettings;
      pendingSettings = {};
      api.putSettings(change).catch((failure) => store.dispatch(errorNotice(failure)));
    }, 300);
  }

  // A change of settings made in another window: this one shows it, and does not send it back.
  function settingsChangedElsewhere(message) {
    if (message.by === CLIENT_ID || sentSettings === null) return;
    const change = {};
    for (const [key, value] of Object.entries(message.settings)) {
      if (key in pendingSettings || key === "language" || sentSettings[key] === value) continue;
      change[key] = value;
    }
    if (Object.keys(change).length === 0) return;
    sentSettings = { ...sentSettings, ...change };
    store.dispatch({ type: "ui.options", change });
    sentSettings = store.getState().options;
  }

  const requested = new Set();
  function loadMissingItems(state) {
    for (const clip of state.clips) {
      if (clip.status !== "analyzed" || clip.items.length > 0 || !clip.analysis?.items || requested.has(clip.id)) continue;
      requested.add(clip.id);
      api.analysis(clip.id).then(
        (analysis) => store.dispatch({ type: "clip.items.loaded", clip_id: clip.id, items: analysis.items }),
        (failure) => store.dispatch(errorNotice(failure)),
      );
    }
  }

  async function loadState() {
    try {
      const [snapshot, settings] = await Promise.all([api.state(), api.settings()]);
      // The language the page is in goes along with every export (the PDF is written in it), also when it was never chosen.
      if (snapshot.languages.includes(saved)) settings.language = saved;
      store.dispatch({ type: "state.loaded", state: snapshot, settings });
      sentSettings = store.getState().options;
    } catch (failure) {
      store.dispatch(errorNotice(failure));
    }
  }

  async function startReview() {
    await api.analyze(store.getState().clips.map((c) => c.id), {});
    await new Promise((resolve) => {
      const stop = store.subscribe((state) => {
        if (state.clips.every((c) => c.status === "analyzed")) {
          stop();
          resolve();
        }
      });
    });
    store.dispatch({ type: "review.open", clip_id: store.getState().clips[0].id });
  }

  // What a reload (or a change of language, which reloads the page) must not lose: the screen and what was typed on Export. Per tab of the
  // browser, never anything about the files; where the browser keeps nothing, the page starts from its defaults as before.
  const VIEW_KEY = "anonymizer.studio.view";
  function savedView() {
    try {
      return JSON.parse(sessionStorage.getItem(VIEW_KEY) ?? "null");
    } catch {
      return null;
    }
  }
  let savedTab = null;
  let savedExport = null;
  let savedSelected = null;
  function saveView(state) {
    if (state.tab === savedTab && state.export === savedExport && state.selectedId === savedSelected) return;
    savedTab = state.tab;
    savedExport = state.export;
    savedSelected = state.selectedId;
    try {
      // The clips taken off the Export list are saved with their names: a Studio started again gives c1 to another file, and the id alone would take that one off.
      const excluded = state.export.excluded.map((id) => ({ id, name: state.clips.find((clip) => clip.id === id)?.name ?? null }));
      const chosen = state.clips.find((clip) => clip.id === state.selectedId);
      sessionStorage.setItem(
        VIEW_KEY,
        JSON.stringify({ tab: state.tab, selected: chosen ? { id: chosen.id, name: chosen.name } : null, export: { options: state.export.options, excluded } }),
      );
    } catch {
      // the browser keeps nothing here: the next load starts from the defaults
    }
  }
  function restoreView() {
    const saved = savedView();
    if (saved) store.dispatch({ type: "ui.restore", saved });
    const state = store.getState();
    // A job that is running says where the person was: its screen. Otherwise the screen they left, when it can be shown.
    const wanted = state.job ? (state.job.kind === "analyze" ? "analyze" : "export") : saved?.tab;
    const tab = tabs.find((candidate) => candidate.id === wanted);
    if (tab && tab.enabled(state)) store.dispatch({ type: "ui.tab", tab: tab.id });
  }

  await loadState();
  if (!mock) restoreView();
  store.subscribe(saveView);
  if (mock) {
    const result = await api.importPaths(["/media/clips/Street shoot"]);
    store.dispatch({ type: "clips.added", clips: result.clips });
    if (params.get("start") === "analyze") api.analyze(store.getState().clips.map((c) => c.id), {});
    if (params.get("start") === "review") await startReview();
    if (params.get("start") === "export") {
      await startReview();
      store.dispatch({ type: "ui.tab", tab: "export" });
    }
    if (params.get("start") === "explorer") document.querySelector(".side-head button").click();
  }
  render(store.getState());
}

start();
