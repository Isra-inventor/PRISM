// Light / night mode for every PRISM page. The choice is stored per browser; without a choice the
// OS setting decides. Switching uses a circular View Transition from the button where supported.
(() => {
  "use strict";
  const KEY = "prism.theme";
  const root = document.documentElement;
  const saved = () => { try { return localStorage.getItem(KEY); } catch (_) { return null; } };
  const effective = () => root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const SUN = '<svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"><circle cx="12" cy="12" r="4.2" fill="none" stroke="currentColor" stroke-width="1.6"/><g stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6"/></g></svg>';
  const MOON = '<svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"><path d="M20 14.6A8.3 8.3 0 0 1 9.4 4a8.3 8.3 0 1 0 10.6 10.6Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';

  function apply(theme) {
    root.dataset.theme = theme;
    try { localStorage.setItem(KEY, theme); } catch (_) {}
    document.querySelectorAll(".theme-toggle").forEach(paint);
    window.dispatchEvent(new CustomEvent("prism-theme", { detail: theme }));
  }
  function paint(btn) {
    const t = effective();
    btn.innerHTML = t === "dark" ? SUN : MOON;
    btn.setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to night mode");
    btn.title = btn.getAttribute("aria-label");
  }
  function toggle(ev) {
    const next = effective() === "dark" ? "light" : "dark";
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!document.startViewTransition || reduce) return apply(next);
    const r = ev.currentTarget.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const end = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    document.startViewTransition(() => apply(next)).ready.then(() => {
      root.animate({ clipPath: [`circle(0 at ${x}px ${y}px)`, `circle(${end}px at ${x}px ${y}px)`] },
        { duration: 520, easing: "cubic-bezier(.2,.7,.2,1)", pseudoElement: "::view-transition-new(root)" });
    }).catch(() => {});
  }
  function mount() {
    const nav = document.querySelector("header.nav");
    if (!nav || nav.querySelector(".theme-toggle")) return;
    const b = document.createElement("button");
    b.type = "button"; b.className = "theme-toggle";
    b.addEventListener("click", toggle);
    paint(b);
    let end = nav.querySelector(".nav-end");
    if (!end) {                       // the nav is a 3-column grid: the right cell holds the actions
      end = document.createElement("div");
      end.className = "nav-end";
      nav.querySelectorAll(".nav-cta").forEach((c) => end.append(c));
      nav.append(end);
    }
    end.append(b);
    matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { if (!saved()) paint(b); });
  }
  if (saved()) root.dataset.theme = saved();
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", mount) : mount();
  window.PRISM_THEME = { effective, apply };
})();
