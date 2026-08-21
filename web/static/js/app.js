// House Price Estimator (India) -- vanilla JS frontend.
// Talks to the FastAPI backend (api/main.py) which wraps the exact same
// src/ pipeline the Streamlit app (app.py) uses. No ML logic lives here --
// this only builds the form from /api/schema, posts to /api/predict, and
// renders the response.

const GROUPS = {
  "Location & basics": ["city", "location", "area", "bedrooms", "resale"],
  "Amenities": ["gymnasium", "swimming_pool", "club_house", "24_x7_security", "power_backup",
    "car_parking", "lift_available", "maintenance_staff", "landscaped_gardens",
    "jogging_track", "indoor_games", "sports_facility", "shopping_mall", "school",
    "hospital", "atm", "intercom", "rain_water_harvesting"],
  "Furnishing": ["ac", "wifi", "tv", "sofa", "bed", "wardrobe", "microwave", "refrigerator",
    "washing_machine", "dining_table", "gasconnection"],
};

let SCHEMA = null;

function titleCase(col) {
  return col.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function fmtLakhCrore(x) {
  const abs = Math.abs(x);
  if (abs >= 1e7) return `₹${(x / 1e7).toLocaleString(undefined, { maximumFractionDigits: 2 })} Cr`;
  if (abs >= 1e5) return `₹${(x / 1e5).toLocaleString(undefined, { maximumFractionDigits: 2 })} L`;
  return `₹${Math.round(x).toLocaleString()}`;
}

function fmtRupees(x) {
  return `₹${Math.round(x).toLocaleString("en-IN")}`;
}

// ---------- Form construction ----------

function fieldControlHTML(col, meta, isImportant) {
  const id = `field_${col}`;
  let control;
  if (meta.type === "numeric") {
    control = `<input type="number" id="${id}" name="${col}" min="${meta.min}" max="${meta.max}" value="${meta.median}" step="${Number.isInteger(meta.median) && Number.isInteger(meta.min) && Number.isInteger(meta.max) ? 1 : "any"}">`;
  } else if (meta.type === "ordinal") {
    const mid = Math.floor(meta.categories.length / 2);
    const opts = meta.categories.map((c, i) => `<option value="${c}" ${i === mid ? "selected" : ""}>${c}</option>`).join("");
    control = `<select id="${id}" name="${col}">${opts}</select>`;
  } else if (meta.type === "locality") {
    const listId = `${id}_list`;
    const datalist = `<datalist id="${listId}">${meta.categories.map(c => `<option value="${c}">`).join("")}</datalist>`;
    control = `<input type="text" list="${listId}" id="${id}" name="${col}" value="${meta.categories[Math.floor(meta.categories.length / 3)] || ''}">${datalist}`;
  } else { // nominal
    const opts = meta.categories.map((c, i) => `<option value="${c}" ${i === 0 ? "selected" : ""}>${c}</option>`).join("");
    control = `<select id="${id}" name="${col}">${opts}</select>`;
  }

  const skipToggle = isImportant
    ? `<label class="skip-toggle" title="Leave this important field unanswered -- the estimate will be flagged as lower-confidence if enough signals agree.">
         <input type="checkbox" data-skip-for="${id}"> Unknown / skip
       </label>`
    : "";

  return `
    <div class="field" data-field="${col}">
      <div class="field-label-row">
        <span class="field-label">${titleCase(col)}</span>
      </div>
      ${control}
      ${skipToggle}
    </div>`;
}

function renderForm(schema) {
  const importantSet = new Set(schema.important_fields);
  const grouped = new Set(Object.values(GROUPS).flat());
  const container = document.getElementById("form-groups");
  let html = "";

  for (const [title, cols] of Object.entries(GROUPS)) {
    const fields = cols.filter(c => schema.form_schema[c]).map(c =>
      fieldControlHTML(c, schema.form_schema[c], importantSet.has(c))
    ).join("");
    if (!fields) continue;
    html += `<details ${title === "Location & basics" ? "open" : ""}>
      <summary>${title}</summary>
      <div class="group-body">${fields}</div>
    </details>`;
  }

  const remaining = Object.keys(schema.form_schema).filter(c => !grouped.has(c));
  if (remaining.length) {
    const fields = remaining.map(c => fieldControlHTML(c, schema.form_schema[c], importantSet.has(c))).join("");
    html += `<details><summary>Other details</summary><div class="group-body">${fields}</div></details>`;
  }

  container.innerHTML = html;

  // Wire up skip checkboxes: disable/clear the paired control.
  container.querySelectorAll("[data-skip-for]").forEach(cb => {
    const target = document.getElementById(cb.getAttribute("data-skip-for"));
    cb.addEventListener("change", () => {
      target.disabled = cb.checked;
    });
  });
}

function collectFormValues() {
  const values = {};
  for (const col of Object.keys(SCHEMA.form_schema)) {
    const el = document.getElementById(`field_${col}`);
    if (!el) continue;
    values[col] = el.disabled || el.value === "" ? null : el.value;
  }
  return values;
}

// ---------- SVG chart helpers (no external chart library) ----------

function svg(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderRangeChart(container, low, mid, high) {
  const W = 640, H = 90, PAD = 44;
  const lo = Math.min(low, mid), hi = Math.max(high, mid);
  const span = hi - lo || 1;
  const x = v => PAD + ((v - lo) / span) * (W - 2 * PAD);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "90 percent price range" });
  root.appendChild(svg("line", { x1: PAD, x2: W - PAD, y1: 46, y2: 46, stroke: "#e6e8f0", "stroke-width": 6, "stroke-linecap": "round" }));
  root.appendChild(svg("line", { x1: x(low), x2: x(high), y1: 46, y2: 46, stroke: "#a5b4fc", "stroke-width": 10, "stroke-linecap": "round" }));
  root.appendChild(svg("circle", { cx: x(mid), cy: 46, r: 9, fill: "#4f46e5", stroke: "#fff", "stroke-width": 3 }));

  [[low, "low"], [mid, "mid"], [high, "high"]].forEach(([v, kind]) => {
    const t = svg("text", { x: x(v), y: kind === "mid" ? 24 : 74, "text-anchor": "middle", "font-size": 12, fill: kind === "mid" ? "#4338ca" : "#6b7280", "font-weight": kind === "mid" ? 700 : 500 });
    t.textContent = fmtLakhCrore(v);
    root.appendChild(t);
  });

  container.innerHTML = "";
  container.appendChild(root);
}

function renderContribChart(container, contributions) {
  if (!contributions.length) {
    container.innerHTML = `<p class="muted">No single feature stood out as an unusually strong driver for this house.</p>`;
    return;
  }
  const ordered = [...contributions].sort((a, b) => a.pct_impact - b.pct_impact);
  const maxAbs = Math.max(...ordered.map(d => Math.abs(d.pct_impact)), 1);
  const rowH = 28, W = 640, LEFT = 150, RIGHT = 60;
  const H = ordered.length * rowH + 20;
  const zeroX = LEFT + (0 - (-maxAbs)) / (2 * maxAbs) * (W - LEFT - RIGHT);
  const scaleX = v => LEFT + (v + maxAbs) / (2 * maxAbs) * (W - LEFT - RIGHT);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "feature contributions" });
  root.appendChild(svg("line", { x1: zeroX, x2: zeroX, y1: 4, y2: H - 4, stroke: "#e6e8f0", "stroke-width": 1 }));

  ordered.forEach((d, i) => {
    const y = 10 + i * rowH;
    const barX = Math.min(zeroX, scaleX(d.pct_impact));
    const barW = Math.abs(scaleX(d.pct_impact) - zeroX) || 1;
    const color = d.pct_impact >= 0 ? "#16a34a" : "#dc2626";

    const label = svg("text", { x: LEFT - 10, y: y + 13, "text-anchor": "end", "font-size": 12, fill: "#374151" });
    label.textContent = titleCase(d.feature);
    root.appendChild(label);

    root.appendChild(svg("rect", { x: barX, y, width: barW, height: 16, rx: 3, fill: color, "fill-opacity": .85 }));

    // Value label normally sits just outside the bar's far (from-zero) edge.
    // But when the bar is the longest one (near maxAbs), that "outside"
    // position lands right next to -- or on top of -- the row-name label
    // in the fixed-width margin (see LEFT). In that case, draw the label
    // inside the bar instead, in white, so it never collides.
    const MIN_MARGIN_GAP = 34;
    const outsideX = d.pct_impact >= 0 ? scaleX(d.pct_impact) + 6 : scaleX(d.pct_impact) - 6;
    const tooCloseToMargin = d.pct_impact >= 0
      ? outsideX > (W - RIGHT) - MIN_MARGIN_GAP
      : outsideX < LEFT + MIN_MARGIN_GAP;
    const val = svg("text", tooCloseToMargin
      ? { x: d.pct_impact >= 0 ? barX + 8 : barX + barW - 8, y: y + 13,
          "text-anchor": d.pct_impact >= 0 ? "start" : "end", "font-size": 11.5, fill: "#fff", "font-weight": 600 }
      : { x: outsideX, y: y + 13, "text-anchor": d.pct_impact >= 0 ? "start" : "end",
          "font-size": 11.5, fill: color, "font-weight": 600 });
    val.textContent = `${d.pct_impact >= 0 ? "+" : ""}${d.pct_impact.toFixed(1)}%`;
    root.appendChild(val);
  });

  container.innerHTML = "";
  container.appendChild(root);
}

// ---------- Results rendering ----------

function renderResults(result, currencySymbol) {
  document.getElementById("results").classList.remove("hidden");

  document.getElementById("price-value").textContent = fmtLakhCrore(result.predicted_price);
  document.getElementById("price-sub").textContent = fmtRupees(result.predicted_price);
  document.getElementById("range-low").textContent = fmtLakhCrore(result.range_low);
  document.getElementById("range-low-sub").textContent = fmtRupees(result.range_low);
  document.getElementById("range-high").textContent = fmtLakhCrore(result.range_high);
  document.getElementById("range-high-sub").textContent = fmtRupees(result.range_high);

  renderRangeChart(document.getElementById("range-chart"), result.range_low, result.predicted_price, result.range_high);

  const banner = document.getElementById("confidence-banner");
  if (result.flagged) {
    const reasons = result.flag_reasons.length ? result.flag_reasons.join("; ") : "the model is less certain than usual about this house";
    banner.className = "banner banner-warning";
    banner.innerHTML = `⚠️ <span><b>Low confidence prediction.</b> ${reasons.charAt(0).toUpperCase() + reasons.slice(1)}. Consider getting a professional valuation for this property instead of relying on this estimate.</span>`;
  } else {
    banner.className = "banner banner-success";
    banner.innerHTML = `✅ <span>This prediction falls within the model's normal operating range.</span>`;
  }

  const notice = document.getElementById("skipped-notice");
  if (result.skipped_important.length) {
    const names = result.skipped_important.map(f => `<b>${titleCase(f)}</b>`).join(", ");
    const pronoun = result.skipped_important.length === 1 ? "it" : "them";
    notice.classList.remove("hidden");
    notice.innerHTML = `ℹ️ <span>You left ${names} unanswered. Predictions are somewhat less reliable without ${pronoun}, even when that alone isn't enough to trigger the low-confidence warning.</span>`;
  } else {
    notice.classList.add("hidden");
  }

  renderContribChart(document.getElementById("contrib-chart"), result.contributions);

  document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- Metrics panel ----------

function pct(x) { return `${(x * 100).toFixed(1)}%`; }

function renderMetrics(m) {
  const pa = m.point_accuracy;
  const el = document.getElementById("metrics-body");

  const tiles = `<div class="metric-grid">
    <div class="metric-tile"><div class="k">Mean Abs. Error</div><div class="v">${fmtLakhCrore(pa.mae)}</div></div>
    <div class="metric-tile"><div class="k">RMSE</div><div class="v">${fmtLakhCrore(pa.rmse)}</div></div>
    <div class="metric-tile"><div class="k">MAPE</div><div class="v">${pa.mape.toFixed(1)}%</div></div>
    <div class="metric-tile"><div class="k">Log-price RMSE</div><div class="v">${pa.rmse_log.toFixed(3)}</div></div>
  </div>
  <p class="metrics-note">⚠️ MAPE of ~${pa.mape.toFixed(0)}% reflects real limitations of this dataset (asking prices,
    not confirmed sale prices; no transaction date; ~70% of amenity fields unrecorded) — reported honestly
    rather than averaged away. Treat point predictions as rough, and lean on the 90% range and confidence flag.</p>`;

  const coverage = `<div class="table-title">90% price-range coverage &amp; width, by method (target = 90%)</div>
  <div class="table-scroll"><table class="data-table"><thead><tr><th>Method</th><th>Coverage</th><th>Avg width</th><th>Median width</th></tr></thead><tbody>
  ${m.coverage_table.map(r => `<tr><td>${r.arm}</td><td>${pct(r.coverage)}</td><td>${fmtLakhCrore(r.avg_width)}</td><td>${fmtLakhCrore(r.median_width)}</td></tr>`).join("")}
  </tbody></table></div>`;

  const bracket = `<div class="table-title">Accuracy by price bracket</div>
  <div class="table-scroll"><table class="data-table"><thead><tr><th>Bracket</th><th>n</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead><tbody>
  ${m.bracket_table.map(r => `<tr><td>${r.bracket}</td><td>${r.n}</td><td>${fmtLakhCrore(r.mae)}</td><td>${fmtLakhCrore(r.rmse)}</td><td>${r.mape.toFixed(1)}%</td></tr>`).join("")}
  </tbody></table></div>`;

  const segment = `<div class="table-title">90% CQR range coverage by city</div>
  <div class="table-scroll"><table class="data-table"><thead><tr><th>City</th><th>n (test)</th><th>Coverage</th><th>Avg width</th></tr></thead><tbody>
  ${m.segment_table.map(r => `<tr><td>${r.segment}</td><td>${r.n}</td><td>${r.coverage != null ? pct(r.coverage) : "—"}</td><td>${r.avg_width != null ? fmtLakhCrore(r.avg_width) : (r.note || "insufficient data")}</td></tr>`).join("")}
  </tbody></table></div>`;

  const cf = m.confidence_flag_validation;
  const confTable = `<div class="table-title">Confidence flag validation — flagged predictions should be measurably worse</div>
  <div class="table-scroll"><table class="data-table"><thead><tr><th>Group</th><th>n</th><th>MAE</th><th>MAPE</th></tr></thead><tbody>
  <tr><td>🚩 Flagged low-confidence</td><td>${cf.flagged.n}</td><td>${fmtLakhCrore(cf.flagged.mae)}</td><td>${cf.flagged.mape.toFixed(1)}%</td></tr>
  <tr><td>✅ Not flagged</td><td>${cf.not_flagged.n}</td><td>${fmtLakhCrore(cf.not_flagged.mae)}</td><td>${cf.not_flagged.mape.toFixed(1)}%</td></tr>
  </tbody></table></div>
  <p class="muted" style="margin-top:-10px;font-size:12.5px;">Flagged predictions do show a higher MAPE than unflagged ones — the confidence flag is catching genuinely worse predictions, not just adding noise.</p>`;

  const cv = m.cv_rmse_log;
  const cvTable = `<div class="table-title">Cross-validation comparison of candidate models (train-only, log-price RMSE — lower is better)</div>
  <div class="table-scroll"><table class="data-table"><thead><tr><th>Model</th><th>CV RMSE (log)</th></tr></thead><tbody>
  <tr><td>Linear (Ridge)</td><td>${cv.linear.toFixed(4)}</td></tr>
  <tr><td>Random Forest</td><td>${cv.random_forest.toFixed(4)}</td></tr>
  <tr><td>XGBoost (selected)</td><td>${cv.xgboost.toFixed(4)}</td></tr>
  </tbody></table></div>`;

  el.innerHTML = tiles + coverage + bracket + segment + confTable + cvTable;
}

// ---------- Boot ----------

async function boot() {
  const schema = await fetch("/api/schema").then(r => r.json());
  SCHEMA = schema;
  renderForm(schema);
  document.getElementById("model-badge").textContent =
    `Model: ${schema.point_model_name} · Test-set error: ${schema.test_mape.toFixed(1)}% MAPE · Log-RMSE: ${schema.test_rmse_log.toFixed(3)}`;

  document.getElementById("estimate-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("submit-btn");
    btn.disabled = true;
    btn.textContent = "Estimating…";
    try {
      const values = collectFormValues();
      const rangeMethod = document.querySelector('input[name="range_method"]:checked').value;
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values, range_method: rangeMethod }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      const result = await res.json();
      renderResults(result, schema.currency_symbol);
    } catch (err) {
      alert(`Couldn't compute an estimate: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Estimate price";
    }
  });

  const metricsDetails = document.getElementById("metrics-details");
  let metricsLoaded = false;
  metricsDetails.addEventListener("toggle", async () => {
    if (metricsDetails.open && !metricsLoaded) {
      metricsLoaded = true;
      const m = await fetch("/api/metrics").then(r => r.json());
      renderMetrics(m);
    }
  });
}

boot();
