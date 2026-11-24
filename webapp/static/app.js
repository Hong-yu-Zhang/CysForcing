(() => {
  const $ = (s) => document.querySelector(s);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fmt = (n) => n.toLocaleString("en-US");
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // probability -> colour, along the cyan of the page: slate below the threshold, bright cyan above it
  const STOPS = [[0, [58, 70, 86]], [0.5, [38, 198, 218]], [1, [217, 251, 255]]];
  const colour = (p) => {
    let i = 0;
    while (i < STOPS.length - 2 && p > STOPS[i + 1][0]) i++;
    const [a, ca] = STOPS[i], [b, cb] = STOPS[i + 1];
    const t = Math.min(1, Math.max(0, (p - a) / (b - a)));
    return "#" + ca.map((v, k) => Math.round(v + (cb[k] - v) * t).toString(16).padStart(2, "0")).join("");
  };

  // ---------------------------------------------------------------- server status
  let device = "GPU";
  const status = $("#status");
  fetch("api/health").then((r) => r.json()).then((h) => {
    device = h.device;
    status.classList.add("ok");
    status.lastElementChild.textContent = "Ready";
  }).catch(() => {
    status.classList.add("down");
    status.lastElementChild.textContent = "Server offline";
  });

  // ---------------------------------------------------------------- input mode, with the sliding pill of the project page
  const seg = $("#mode");
  const tabs = [...seg.querySelectorAll("button")];
  let mode = "uniprot";
  const placeThumb = () => {
    const b = tabs.find((t) => t.dataset.mode === mode);
    seg.style.setProperty("--x", b.offsetLeft + "px");
    seg.style.setProperty("--w", b.offsetWidth + "px");
  };
  const setMode = (m) => {
    mode = m;
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.mode === m)));
    document.querySelectorAll(".pane").forEach((p) => { p.hidden = p.dataset.pane !== m; });
    placeThumb();
    hideError();
  };
  tabs.forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
  addEventListener("resize", placeThumb);
  if (document.fonts) document.fonts.ready.then(placeThumb);
  placeThumb();

  const acc = $("#acc");
  document.querySelectorAll("[data-acc]").forEach((b) => b.addEventListener("click", () => {
    acc.value = b.dataset.acc;
    $("#form").requestSubmit();
  }));

  let upload = null;
  const drop = $("#drop"), file = $("#file"), fname = $("#fname");
  const takeFile = async (f) => {
    if (!f) return;
    if (f.size > 12 * 1024 * 1024) return showError("This file is larger than 12 MB.");
    upload = { name: f.name, text: await f.text() };
    drop.classList.add("has");
    drop.querySelector("b").textContent = f.name;
    fname.textContent = (f.size / 1024).toFixed(0) + " KB · click to choose another";
    hideError();
  };
  file.addEventListener("change", () => takeFile(file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); takeFile(e.dataTransfer.files[0]); });

  // ---------------------------------------------------------------- run
  const err = $("#error");
  const showError = (m) => { err.textContent = m; err.hidden = false; };
  const hideError = () => { err.hidden = true; };
  const run = $("#run"), busy = $("#busy"), clock = $("#clock");

  $("#form").addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    let body, pdbText = null;
    if (mode === "uniprot") {
      const a = acc.value.trim().toUpperCase();
      if (!a) return showError("Type a UniProt accession, or pick an example.");
      body = { uniprot: a };
      $("#busy-msg").textContent = "Fetching the AlphaFold DB model and scoring on the " + device;
    } else {
      if (!upload) return showError("Choose a PDB file first.");
      body = { pdb: upload.text, name: upload.name };
      pdbText = upload.text;
      $("#busy-msg").textContent = "Scoring on the " + device;
    }
    run.disabled = true;
    busy.hidden = false;
    const t0 = performance.now();
    const tick = setInterval(() => { clock.textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s"; }, 100);
    try {
      const r = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const res = await r.json().catch(() => ({ error: "The server sent an unreadable reply (HTTP " + r.status + ")." }));
      if (!r.ok || res.error) throw new Error(res.error || "HTTP " + r.status);
      busy.hidden = true;        // before the scroll to the results, or they shift up under the bar as it closes
      render(res, res.pdb || pdbText);
    } catch (ex) {
      showError(ex.message === "Failed to fetch" ? "The server could not be reached." : ex.message);
    } finally {
      clearInterval(tick);
      run.disabled = false;
      busy.hidden = true;
    }
  });

  // ---------------------------------------------------------------- results
  let current = null, sortKey = "prob", selected = null;
  const tip = $("#tip");

  function render(res, pdbText) {
    current = res;
    selected = null;
    const m = res.meta || {};
    if (res.source === "AlphaFold DB") {
      $("#r-source").textContent = ["AlphaFold DB", m.entry, m.version ? "v" + m.version : null].filter(Boolean).join(" · ");
      $("#r-name").textContent = m.description || m.entry || res.name;
      $("#r-meta").innerHTML = [m.gene && "<i>" + esc(m.gene) + "</i>", m.organism && esc(m.organism)].filter(Boolean).join(" · ");
    } else {
      $("#r-source").textContent = "Uploaded structure";
      $("#r-name").textContent = res.name;
      $("#r-meta").textContent = "Sequence from " + res.sequence_source + " · chain A";
    }
    const scored = res.sites.filter((s) => s.prob !== null);
    const pos = scored.filter((s) => s.call);
    $("#f-len").textContent = fmt(res.length);
    $("#f-len").nextElementSibling.textContent = res.length > res.scored_length ? "residues · first " + fmt(res.scored_length) + " scored" : "residues";
    $("#f-cys").textContent = fmt(res.sites.length);
    $("#f-pos").textContent = fmt(pos.length);
    $("#f-time").textContent = res.seconds.toFixed(2) + " s";
    $("#f-dev").textContent = "on the " + res.device;

    const box = $("#results");
    box.hidden = false;
    drawTrack(res);
    drawRows();
    box.style.animation = "none";
    void box.offsetWidth;
    box.style.animation = "";
    box.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    drawViewer(pdbText, res.sites);
  }

  function drawTrack(res) {
    // one SVG unit per CSS pixel, so the labels keep their size on a phone
    const W = Math.max(300, Math.round($("#track").clientWidth) || 1100), H = W < 600 ? 170 : 200;
    const l = 44, r = 14, t = 14, b = 30;
    const L = Math.max(2, res.length);
    const x = (p) => l + (p - 1) / (L - 1) * (W - l - r);
    const y = (p) => t + (1 - p) * (H - t - b);
    const raw = L / (W < 600 ? 4 : 8), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Probability of each cysteine along the sequence">`;
    [0.25, 0.5, 0.75, 1].forEach((g) => {
      s += `<line class="${g === 0.5 ? "thr" : "grid"}" x1="${l}" x2="${W - r}" y1="${y(g)}" y2="${y(g)}"/>`;
      s += `<text class="tick" x="${l - 8}" y="${y(g) + 4}" text-anchor="end">${g.toFixed(g === 1 ? 0 : 2)}</text>`;
    });
    s += `<line class="axis" x1="${l}" x2="${W - r}" y1="${y(0)}" y2="${y(0)}"/>`;
    for (let p = step; p <= L; p += step) s += `<text class="tick" x="${x(p)}" y="${H - 8}" text-anchor="middle">${fmt(p)}</text>`;
    s += `<text class="tick" x="${x(1)}" y="${H - 8}" text-anchor="middle">1</text>`;
    if (res.length > res.scored_length) {
      s += `<rect class="ns" x="${x(res.scored_length)}" y="${t}" width="${x(L) - x(res.scored_length)}" height="${y(0) - t}" rx="4"/>`;
    }
    res.sites.forEach((c) => {
      if (c.prob === null) {
        s += `<circle class="dot" data-pos="${c.pos}" cx="${x(c.pos)}" cy="${y(0)}" r="3.5" fill="none" stroke="#66696e"/>`;
        return;
      }
      const col = colour(c.prob);
      s += `<line class="stem" x1="${x(c.pos)}" x2="${x(c.pos)}" y1="${y(0)}" y2="${y(c.prob)}" stroke="${col}" opacity=".55"/>`;
      s += `<circle class="dot" data-pos="${c.pos}" cx="${x(c.pos)}" cy="${y(c.prob)}" r="5" fill="${col}"${c.call ? ' style="filter:drop-shadow(0 0 4px #26c6da)"' : ""}/>`;
    });
    s += "</svg>";
    const track = $("#track");
    track.innerHTML = s;
    track.querySelectorAll(".dot").forEach((d) => {
      const site = res.sites.find((c) => c.pos === Number(d.dataset.pos));
      d.addEventListener("mouseenter", (e) => showTip(e, site));
      d.addEventListener("mousemove", moveTip);
      d.addEventListener("mouseleave", () => { tip.hidden = true; });
      d.addEventListener("click", () => select(site.pos, true));
    });
  }

  function showTip(e, c) {
    const p = c.prob === null ? "not scored (beyond residue " + fmt(current.scored_length) + ")"
      : "p = " + c.prob.toFixed(3) + (c.call ? " · reactive" : "");
    tip.innerHTML = `<strong>C${c.pos}</strong>${esc(p)}<br><span>${esc(c.left)}<b style="color:#78e4f0">C</b>${esc(c.right)}</span>`;
    tip.hidden = false;
    moveTip(e);
  }
  function moveTip(e) {
    tip.style.left = Math.min(innerWidth - tip.offsetWidth - 12, e.clientX + 14) + "px";
    tip.style.top = e.clientY - tip.offsetHeight - 12 + "px";
  }

  function drawRows() {
    const sites = [...current.sites].sort(sortKey === "pos" ? (a, b) => a.pos - b.pos
      : (a, b) => (b.prob ?? -1) - (a.prob ?? -1) || a.pos - b.pos);
    $("#rows").innerHTML = sites.map((c) => {
      const pl = c.plddt === null ? "—" : c.plddt.toFixed(0);
      const prob = c.prob === null
        ? `<div class="prob na"><b>not scored</b></div>`
        : `<div class="prob ${c.call ? "pos" : "neg"}"><div class="pbar"><span style="width:${(100 * c.prob).toFixed(1)}%;background:${colour(c.prob)}"></span></div><b>${c.prob.toFixed(3)}</b></div>`;
      return `<tr data-pos="${c.pos}"${c.pos === selected ? ' class="sel"' : ""}><td class="site">C${c.pos}</td>` +
        `<td class="ctx">${esc(c.left.padStart(7, "·"))}<b>C</b>${esc(c.right.padEnd(7, "·"))}</td><td class="pl">${pl}</td><td>${prob}</td></tr>`;
    }).join("");
    $("#rows").querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => select(Number(tr.dataset.pos), false)));
  }
  $("#sort").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    sortKey = b.dataset.sort;
    $("#sort").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    drawRows();
  }));

  // ---------------------------------------------------------------- 3D viewer
  let viewer = null, selLabel = null, hoverLabel = null;
  function drawViewer(pdbText, sites) {
    const el = $("#viewer"), msg = $("#viewer-msg");
    if (viewer) { viewer.clear(); }
    if (!window.$3Dmol) { msg.textContent = "The 3D viewer could not be loaded (3Dmol.js is fetched from jsDelivr)."; return; }
    if (!pdbText) { msg.textContent = "No structure to show."; return; }
    msg.hidden = true;
    if (!viewer) viewer = $3Dmol.createViewer(el, { backgroundColor: "#050b12", antialias: true });
    viewer.addModel(pdbText, "pdb");
    viewer.setStyle({}, {});
    viewer.setStyle({ chain: "A" }, { cartoon: { color: "#3b4f63" } });
    sites.forEach((c) => {
      if (c.prob === null) return;
      const col = colour(c.prob);
      viewer.addStyle({ chain: "A", resi: c.pos }, { stick: { radius: 0.16, color: col } });
      viewer.addStyle({ chain: "A", resi: c.pos, atom: "SG" }, { sphere: { radius: 0.9 + 0.9 * c.prob, color: col } });
    });
    const byPos = new Map(sites.map((c) => [c.pos, c]));
    viewer.setHoverable({ chain: "A", atom: "SG" }, true, (atom) => {
      const c = byPos.get(atom.resi);
      if (!c || hoverLabel) return;
      hoverLabel = viewer.addLabel(`C${c.pos}  ${c.prob === null ? "—" : c.prob.toFixed(2)}`, labelStyle(atom, "#46596d"));
      viewer.render();
    }, () => {
      if (hoverLabel) { viewer.removeLabel(hoverLabel); hoverLabel = null; viewer.render(); }
    });
    viewer.setClickable({ chain: "A", atom: "SG" }, true, (atom) => select(atom.resi, true));
    viewer.zoomTo({ chain: "A" });
    viewer.render();
  }
  const labelStyle = (atom, border) => ({
    position: { x: atom.x, y: atom.y, z: atom.z }, backgroundColor: "#0b131d", backgroundOpacity: 0.92,
    borderColor: border, borderThickness: 1, fontColor: "#eeeff1", fontSize: 13, font: "Google Sans", inFront: true,
  });

  function select(pos, fromPlot) {
    selected = pos;
    document.querySelectorAll("#rows tr").forEach((tr) => tr.classList.toggle("sel", Number(tr.dataset.pos) === pos));
    document.querySelectorAll("#track .dot").forEach((d) => d.classList.toggle("sel", Number(d.dataset.pos) === pos));
    if (fromPlot) {
      const tr = document.querySelector(`#rows tr[data-pos="${pos}"]`);
      if (tr) tr.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
    }
    if (!viewer) return;
    const atom = viewer.selectedAtoms({ chain: "A", resi: pos, atom: "SG" })[0];
    if (selLabel) { viewer.removeLabel(selLabel); selLabel = null; }
    if (!atom) return;
    const c = current.sites.find((s) => s.pos === pos);
    selLabel = viewer.addLabel(`C${pos}  ${c.prob === null ? "—" : c.prob.toFixed(3)}`, labelStyle(atom, "#26c6da"));
    viewer.zoomTo({ within: { distance: 12, sel: { chain: "A", resi: pos } } }, reduce ? 0 : 700);
    viewer.render();
  }

  let resizeTimer = null;
  addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!current) return;
      drawTrack(current);
      document.querySelectorAll("#track .dot").forEach((d) => d.classList.toggle("sel", Number(d.dataset.pos) === selected));
      if (viewer) { viewer.resize(); viewer.render(); }
    }, 150);
  });

  // ---------------------------------------------------------------- downloads
  const save = (text, name, type) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  const stem = () => current.name.replace(/\.(pdb|ent)$/i, "");
  $("#dl-json").addEventListener("click", () => save(JSON.stringify(current.result_json, null, 4), stem() + "_result.json", "application/json"));
  $("#dl-csv").addEventListener("click", () => {
    const rows = [["site", "position", "probability", "reactive_at_0.5", "plddt", "context"]];
    current.sites.forEach((c) => rows.push([`C${c.pos}`, c.pos, c.prob === null ? "" : c.prob, c.call === null ? "" : c.call,
      c.plddt === null ? "" : c.plddt, c.left + "C" + c.right]));
    save(rows.map((r) => r.join(",")).join("\n") + "\n", stem() + "_cysteines.csv", "text/csv");
  });
})();
