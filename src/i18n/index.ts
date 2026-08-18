import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./en.json";
import tr from "./tr.json";

const STORAGE_KEY = "cayahub.language";

export const LANGUAGES = [
  { code: "tr", label: "Türkçe" },
  { code: "en", label: "English" },
] as const;

export type LanguageCode = (typeof LANGUAGES)[number]["code"];

/**
 * Turkish is the default because that is the audience this was built for.
 * A stored choice always wins; otherwise the browser's language decides, so a
 * non-Turkish user is not dropped into a language they cannot read.
 */
function initialLanguage(): LanguageCode {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "tr" || stored === "en") return stored;
  return navigator.language?.toLowerCase().startsWith("en") ? "en" : "tr";
}

i18next.use(initReactI18next).init({
  resources: { en: { translation: en }, tr: { translation: tr } },
  lng: initialLanguage(),
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

export function setLanguage(code: LanguageCode): void {
  localStorage.setItem(STORAGE_KEY, code);
  i18next.changeLanguage(code);
}

export default i18next;
