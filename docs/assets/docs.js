/* GradieX docs — theme toggle, nav filter, on-this-page tracking, copy buttons. */
(function () {
  "use strict";

  // ---- theme -------------------------------------------------------------
  var KEY = "gradiex-theme";
  try {
    var saved = localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") {
      document.documentElement.setAttribute("data-theme", saved);
    }
  } catch (e) { /* private mode: fall back to the OS preference */ }

  function currentTheme() {
    var set = document.documentElement.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest("[data-theme-toggle]");
    if (!btn) return;
    var next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
    btn.setAttribute("aria-label", "Switch to " + (next === "dark" ? "light" : "dark") + " theme");
  });

  // ---- version selector --------------------------------------------------
  var vs = document.querySelector("[data-version-select]");
  if (vs) {
    vs.addEventListener("change", function () {
      if (vs.value) window.location.href = vs.value;
    });
  }

  // ---- nav filter --------------------------------------------------------
  document.querySelectorAll("[data-nav-filter]").forEach(function (input) {
    var scope = input.closest("[data-nav-scope]") || document;
    input.addEventListener("input", function () {
      var q = input.value.trim().toLowerCase();
      scope.querySelectorAll(".nav-group").forEach(function (group) {
        var shown = 0;
        group.querySelectorAll(".nav-list li").forEach(function (li) {
          var hit = !q || li.textContent.toLowerCase().indexOf(q) !== -1;
          li.hidden = !hit;
          if (hit) shown++;
        });
        group.hidden = q && shown === 0;
      });
      var any = scope.querySelectorAll(".nav-group:not([hidden])").length;
      var empty = scope.querySelector(".nav-empty");
      if (empty) empty.hidden = !(q && any === 0);
    });
  });

  // ---- copy buttons ------------------------------------------------------
  document.querySelectorAll(".content pre").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var code = pre.querySelector("code");
      var text = code ? code.innerText : pre.innerText;
      var done = function () {
        btn.textContent = "Copied";
        btn.classList.add("done");
        setTimeout(function () {
          btn.textContent = "Copy";
          btn.classList.remove("done");
        }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { btn.textContent = "Failed"; });
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { btn.textContent = "Failed"; }
        document.body.removeChild(ta);
      }
    });
    pre.appendChild(btn);
  });

  // ---- on-this-page tracking --------------------------------------------
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc-list a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    var targets = [];
    tocLinks.forEach(function (a) {
      var el = document.getElementById(a.getAttribute("href").slice(1));
      if (el) { byId[el.id] = a; targets.push(el); }
    });
    var visible = new Set();
    var mark = function () {
      if (!visible.size) return;
      var first = targets.filter(function (t) { return visible.has(t.id); })[0];
      if (!first) return;
      tocLinks.forEach(function (a) { a.classList.remove("active"); });
      if (byId[first.id]) byId[first.id].classList.add("active");
    };
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) visible.add(e.target.id);
        else visible.delete(e.target.id);
      });
      mark();
    }, { rootMargin: "-15% 0px -70% 0px", threshold: 0 });
    targets.forEach(function (t) { io.observe(t); });
    if (tocLinks[0]) tocLinks[0].classList.add("active");
  }
})();
