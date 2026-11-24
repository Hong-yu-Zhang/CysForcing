(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const bar = document.getElementById("progress");
  const onScroll = () => {
    const h = document.documentElement;
    bar.style.width = (100 * h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight)).toFixed(2) + "%";
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }), { threshold: 0.15 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  // Bar at the top: the pill follows the chapter in view, and the wordmark appears once the title has
  // scrolled under the bar.
  const nav = document.getElementById("topnav");
  const seg = document.getElementById("seg");
  const segLinks = [...seg.querySelectorAll("a")];
  const chapters = segLinks.map((a) => document.getElementById(a.getAttribute("href").slice(1)));
  const title = document.querySelector(".title");
  let current = -2;
  const placeThumb = () => {
    const a = segLinks[current];
    if (!a) return;
    seg.style.setProperty("--x", a.offsetLeft + "px");
    seg.style.setProperty("--w", a.offsetWidth + "px");
  };
  const spy = () => {
    const line = nav.getBoundingClientRect().bottom + 28;
    let idx = -1;
    chapters.forEach((c, i) => {
      const r = c.getBoundingClientRect();
      if (r.top <= line && r.bottom > line) idx = i;
    });
    nav.classList.toggle("at-top", title.getBoundingClientRect().bottom > line - 28);
    if (idx === current) return;
    current = idx;
    segLinks.forEach((a, i) => {
      if (i === idx) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
    seg.classList.toggle("has-active", idx >= 0);
    if (idx < 0) return;
    placeThumb();
    if (seg.scrollWidth > seg.clientWidth) {   // on a phone the control scrolls sideways: keep the pill in view
      seg.scrollLeft = segLinks[idx].offsetLeft - (seg.clientWidth - segLinks[idx].offsetWidth) / 2;
    }
  };
  let spyQueued = false;
  const queueSpy = () => {
    if (spyQueued) return;
    spyQueued = true;
    requestAnimationFrame(() => { spyQueued = false; spy(); });
  };
  addEventListener("scroll", queueSpy, { passive: true });
  segLinks.forEach((a, i) => a.addEventListener("click", (e) => {
    e.preventDefault();
    chapters[i].scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }));
  nav.querySelector(".brand").addEventListener("click", (e) => {
    e.preventDefault();
    scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
  });

  // Metric tabs: the buttons sit over the tabs drawn in each frame, and a ring slides to the one shown;
  // while in view they advance on their own, as in the video, until the reader picks one.
  const rings = [];
  document.querySelectorAll(".tabs").forEach((box) => {
    const imgs = [...box.querySelectorAll("img")];
    const btns = [...box.querySelectorAll("button")];
    const ring = box.querySelector(".tab-ind");
    let k = 0, timer = null, manual = false;
    const place = () => {
      const b = btns[k];
      if (!ring || !b || !b.offsetWidth) return;
      ring.style.setProperty("--x", b.offsetLeft + "px");
      ring.style.setProperty("--w", b.offsetWidth + "px");
      ring.style.setProperty("--t", b.offsetTop + "px");
      ring.style.setProperty("--h", b.offsetHeight + "px");
      box.classList.add("ready");
    };
    const show = (i) => { imgs[k].classList.remove("on"); k = i; imgs[k].classList.add("on"); place(); };
    rings.push(place);
    imgs[0].complete ? place() : imgs[0].addEventListener("load", place, { once: true });
    btns.forEach((b, i) => b.addEventListener("click", () => {
      manual = true;
      clearInterval(timer);
      show(i);
    }));
    const period = Number(box.dataset.auto) || 0;
    if (!period || reduce) return;
    new IntersectionObserver(([e]) => {
      if (manual) return;
      if (e.isIntersecting && !timer) timer = setInterval(() => show((k + 1) % imgs.length), period);
      if (!e.isIntersecting && timer) { clearInterval(timer); timer = null; }
    }, { threshold: 0.5 }).observe(box);
  });

  addEventListener("resize", () => { placeThumb(); queueSpy(); rings.forEach((f) => f()); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { placeThumb(); rings.forEach((f) => f()); });
  spy();

  // Summary numbers count up once, like the summary scene of the video; they start at zero so the
  // final value does not flash before the count begins.
  const ease = (x) => 1 - Math.pow(1 - x, 3);
  const zero = (el) => (el.dataset.pre || "") + (0).toFixed(Number(el.dataset.dec)) + (el.dataset.post || "");
  if (!reduce) document.querySelectorAll("[data-to]").forEach((el) => { el.textContent = zero(el); });
  const counters = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (!e.isIntersecting) return;
    counters.unobserve(e.target);
    if (reduce) return;
    const el = e.target, to = Number(el.dataset.to), dec = Number(el.dataset.dec);
    const pre = el.dataset.pre || "", post = el.dataset.post || "", t0 = performance.now();
    const step = (now) => {
      const u = Math.min(1, (now - t0) / 1200);
      el.textContent = pre + (to * ease(u)).toFixed(dec) + post;
      if (u < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }), { threshold: 0.6 });
  document.querySelectorAll("[data-to]").forEach((el) => counters.observe(el));

  document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.querySelector(b.dataset.copy).textContent);
      b.textContent = "Copied";
      b.classList.add("done");
    } catch (err) {
      b.textContent = "Select and copy";
    }
    setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1800);
  }));
})();
