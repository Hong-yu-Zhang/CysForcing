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

  // Metric tabs: the buttons sit over the tabs drawn in each frame; while in view they advance
  // on their own, as in the video, until the reader picks one.
  document.querySelectorAll(".tabs").forEach((box) => {
    const imgs = [...box.querySelectorAll("img")];
    let k = 0, timer = null, manual = false;
    const show = (i) => { imgs[k].classList.remove("on"); k = i; imgs[k].classList.add("on"); };
    box.querySelectorAll("button").forEach((b, i) => b.addEventListener("click", () => {
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

  if (reduce) document.querySelectorAll("video[autoplay]").forEach((v) => v.pause());
})();
