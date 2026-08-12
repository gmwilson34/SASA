/* ============================================================================
   SASA — pre-paint bootstrap
   Ridgeback Defense

   This has to run BEFORE the first paint, which is why it is a separate file
   loaded synchronously in <head> rather than a <script> block inside the
   document. The application's own Content-Security-Policy is

       script-src 'self'

   with no 'unsafe-inline' and no hash, so an inline block here is refused by
   the browser and silently does nothing — which is exactly what was happening:
   the guard existed, the console reported the refusal, and a dark-theme
   operator got a white flash on every launch anyway.

   Nothing here may depend on the DOM. It reads two keys and stamps two
   attributes on <html>, which is the only element that exists at this point.

     sasa.theme    "light" | "dark" | "system"   -> data-theme
     sasa.sidebar  "full"  | "rail"              -> data-sidebar

   LIGHT IS THE DEFAULT, not "system". Measurement output is printed and
   shared, and a report should look on screen the way it looks on paper; a
   machine that happens to be set to dark should not change what a technician
   sees the first time they open the application. Only "system", chosen
   explicitly in Settings, leaves data-theme off and lets the
   prefers-color-scheme block in tokens.css decide.
   ============================================================================ */

(function () {
  var root = document.documentElement;

  try {
    var theme = localStorage.getItem('sasa.theme');
    root.setAttribute('data-theme', theme === 'dark' ? 'dark'
      : theme === 'system' ? '' : 'light');
    if (theme === 'system') root.removeAttribute('data-theme');
  } catch (e) {
    root.setAttribute('data-theme', 'light');
  }

  try {
    // Unquoted for historical reasons: this key is written with setRaw, not
    // JSON.stringify, so it may be either "rail" or "\"rail\"".
    var sidebar = localStorage.getItem('sasa.sidebar');
    if (sidebar) sidebar = sidebar.replace(/^"|"$/g, '');
    root.setAttribute('data-sidebar', sidebar === 'rail' ? 'rail' : 'full');
  } catch (e) {
    root.setAttribute('data-sidebar', 'full');
  }
})();
