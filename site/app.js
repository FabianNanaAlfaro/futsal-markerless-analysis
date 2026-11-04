const themeButton = document.querySelector("#theme-toggle");
const themeColor = document.querySelector('meta[name="theme-color"]');
const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
const themeStorageKey = "futsal-markerless-analysis-theme";

function getStoredTheme() {
  try {
    const saved = window.localStorage.getItem(themeStorageKey);
    return saved === "light" || saved === "dark" ? saved : null;
  } catch {
    return null;
  }
}

function applyTheme(theme, savePreference = false) {
  const selectedTheme = theme === "light" || theme === "dark" ? theme : null;
  const resolvedTheme = selectedTheme || (systemTheme.matches ? "dark" : "light");
  if (selectedTheme) {
    document.documentElement.dataset.theme = selectedTheme;
  } else {
    delete document.documentElement.dataset.theme;
  }
  if (savePreference) {
    try {
      window.localStorage.setItem(themeStorageKey, resolvedTheme);
    } catch {
      // The selected appearance still applies for this page view.
    }
  }

  themeButton.textContent = resolvedTheme === "dark" ? "Light mode" : "Dark mode";
  themeButton.setAttribute(
    "aria-label",
    "Switch to " + (resolvedTheme === "dark" ? "light" : "dark") + " mode",
  );
  themeButton.setAttribute("aria-pressed", String(resolvedTheme === "dark"));
  themeColor.setAttribute("content", resolvedTheme === "dark" ? "#151a1c" : "#fbfaf7");
}

applyTheme(getStoredTheme());
themeButton.addEventListener("click", () => {
  const currentTheme =
    getStoredTheme() ||
    document.documentElement.dataset.theme ||
    (systemTheme.matches ? "dark" : "light");
  applyTheme(currentTheme === "dark" ? "light" : "dark", true);
});
systemTheme.addEventListener("change", () => {
  if (!getStoredTheme()) applyTheme(null);
});
