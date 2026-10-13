export function createI18n(dict, locale = "en") {
  const plural = new Intl.PluralRules(locale);

  function lookup(key, params) {
    if (params && typeof params.count === "number") {
      const form = dict[`${key}.${plural.select(params.count)}`];
      if (form !== undefined) return form;
    }
    return dict[key];
  }

  function t(key, params = {}) {
    // The limit that a file went over is named by the server in a word of its own ("photos", "clips", "seconds", "height"): said in the
    // page's language, not in English inside a German sentence.
    if (key === "error.demo.limit" && typeof params.limit === "string" && dict[`demo.limit.${params.limit}`] !== undefined) {
      params = { ...params, limit: dict[`demo.limit.${params.limit}`] };
    }
    // "That cannot be done with this file now" says what the file is doing when the server knows: still being analyzed, being exported, never analyzed.
    if (key === "error.clip.state" && typeof params.status === "string" && dict[`error.clip.state.${params.status}`] !== undefined) key = `error.clip.state.${params.status}`;
    const text = lookup(key, params);
    if (text === undefined) return `[${key}]`;
    return text.replace(/\{(\w+)\}/g, (whole, name) => (name in params ? String(params[name]) : whole));
  }

  return { t, locale };
}

// Names shown in the language menu, each in its own language (escapes keep this file plain ASCII).
export const LANGUAGE_NAMES = {
  en: "English", es: "Espa\u00f1ol", fr: "Fran\u00e7ais", de: "Deutsch", pt: "Portugu\u00eas", it: "Italiano", "zh-Hans": "\u7b80\u4f53\u4e2d\u6587", pseudo: "Pseudo +40%",
};

const ACCENTS = { a: "\u00e0", e: "\u00e9", i: "\u00ee", o: "\u00f6", u: "\u00fc", c: "\u00e7", n: "\u00f1", A: "\u00c5", E: "\u00c9", I: "\u00ce", O: "\u00d6", U: "\u00dc" };

// A test language for layout: accented letters, 40% longer, placeholders untouched. It shows what a longer
// translation does to every control without needing one.
export function pseudoText(text) {
  const parts = text.split(/(\{\w+\})/);
  const body = parts.map((part) => (/^\{\w+\}$/.test(part) ? part : part.replace(/[aeioucnAEIOU]/g, (c) => ACCENTS[c]))).join("");
  const extra = Math.ceil(text.replace(/\{\w+\}/g, "").length * 0.4);
  const filler = "lorem ipsum dolor sit amet ".repeat(Math.ceil(extra / 5)).slice(0, extra).trim();
  return filler ? `[${body} ${filler}]` : `[${body}]`;
}

export function pseudoDictionary(dict) {
  return Object.fromEntries(Object.entries(dict).map(([key, text]) => [key, pseudoText(text)]));
}

export async function loadI18n(locale) {
  if (locale === "pseudo") {
    const response = await fetch("i18n/en.json");
    if (response.ok) return createI18n(pseudoDictionary(await response.json()), "en");
  }
  for (const tryLocale of locale === "en" ? ["en"] : [locale, "en"]) {
    const response = await fetch(`i18n/${tryLocale}.json`);
    if (response.ok) return createI18n(await response.json(), tryLocale);
  }
  throw new Error("cannot load the language files");
}
