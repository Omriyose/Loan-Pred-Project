/* ==========================================================
   Loan Eligibility Dashboard – front-end logic
   Vanilla JS only. All data comes from the Flask API.
   ========================================================== */

"use strict";

/* ---------- Small helpers ---------- */

/** Shorthand for document.getElementById */
const $ = (id) => document.getElementById(id);

/**
 * GET a JSON endpoint. Throws if the response is not OK,
 * using the API's {"error": "..."} message when available.
 */
async function getJSON(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return data;
}

/** Create an element with an optional class and text content */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

/** Format a fraction (0.7837) as a percentage string ("78.4%") */
const toPercent = (fraction) => `${(fraction * 100).toFixed(1)}%`;

/** Format a number with thousands separators; null/undefined becomes an em dash */
function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
}

/** Show an error message inside a widget container */
function showWidgetError(container, message) {
  container.replaceChildren(el("p", "widget-error", message));
}

/** Add the entrance animation to an element with a staggered delay */
function animateIn(node, index) {
  node.classList.add("animate-in");
  node.style.animationDelay = `${index * 70}ms`;
}

/* ---------- Startup ---------- */

document.addEventListener("DOMContentLoaded", init);

/**
 * Check the model is ready. If not (or the request fails),
 * show only the "not ready" screen; otherwise load the dashboard.
 */
async function init() {
  let ready = false;
  try {
    const data = await getJSON("/api/check");
    ready = data && data.ready === true;
  } catch (err) {
    ready = false;
  }

  $("loading-screen").classList.add("hidden");

  if (!ready) {
    $("not-ready-screen").classList.remove("hidden");
    return;
  }

  $("dashboard").classList.remove("hidden");

  // Stagger the entrance animation of the dashboard cards
  document.querySelectorAll("#dashboard .card").forEach((card, i) => animateIn(card, i + 4));

  // Load all widgets in parallel; each one handles its own errors
  loadMetrics();
  loadFeatures();
  loadSamples();
  loadInfo();

  $("predict-form").addEventListener("submit", handlePredict);
}

/* ---------- 1 & 2. Metrics + confusion matrix ---------- */

async function loadMetrics() {
  const row = $("metrics-row");
  try {
    const data = await getJSON("/api/metrics");
    renderMetricCards(row, data);
    renderConfusionMatrix($("cm-grid"), data["Confusion Matrix"]);
  } catch (err) {
    showWidgetError(row, `Could not load metrics: ${err.message}`);
    showWidgetError($("cm-grid"), `Could not load confusion matrix: ${err.message}`);
  }
}

/** Render one glass card per metric, with a percentage and a progress bar */
function renderMetricCards(container, data) {
  const keys = ["Accuracy", "Precision", "Recall", "F1 Score"];
  container.replaceChildren();

  keys.forEach((key, i) => {
    const value = data[key];
    const card = el("div", "glass metric-card");
    card.appendChild(el("div", "metric-label", key));
    card.appendChild(el("div", "metric-value", typeof value === "number" ? toPercent(value) : "—"));

    const bar = el("div", "metric-bar");
    const fill = el("div", "metric-bar-fill");
    bar.appendChild(fill);
    card.appendChild(bar);

    animateIn(card, i);
    container.appendChild(card);

    // Grow the bar after the card is on screen
    if (typeof value === "number") {
      requestAnimationFrame(() => {
        setTimeout(() => { fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`; }, 150 + i * 70);
      });
    }
  });
}

/**
 * Draw the 2x2 confusion matrix.
 * matrix[actual][predicted], order N then Y, Y is positive:
 *   [[TN, FP],
 *    [FN, TP]]
 * Each cell is shaded by its value relative to the largest cell.
 */
function renderConfusionMatrix(grid, matrix) {
  grid.replaceChildren();

  const classes = ["N", "Y"];
  const tags = [["TN", "FP"], ["FN", "TP"]];
  const max = Math.max(...matrix.flat(), 1);

  // Header row: blank corner, then predicted labels
  grid.appendChild(el("div", "cm-head", ""));
  classes.forEach((c) => grid.appendChild(el("div", "cm-head", c)));

  matrix.forEach((row, r) => {
    // Actual label at the start of each row
    grid.appendChild(el("div", "cm-head", classes[r]));

    row.forEach((value, c) => {
      const cell = el("div", "cm-cell");
      // Shade: opacity scales from 0.08 (zero) to 0.85 (max)
      const alpha = 0.08 + (value / max) * 0.77;
      cell.style.background = `rgba(124, 140, 255, ${alpha.toFixed(3)})`;
      cell.title = `Actual ${classes[r]}, predicted ${classes[c]}: ${value}`;

      cell.appendChild(el("span", "cm-tag", tags[r][c]));
      cell.appendChild(el("span", "cm-value", String(value)));
      grid.appendChild(cell);
    });
  });
}

/* ---------- 3. Features panel ---------- */

async function loadFeatures() {
  try {
    const data = await getJSON("/api/features");
    fillList($("number-features"), data["Number Features"]);
    fillList($("text-features"), data["Text Features"]);
  } catch (err) {
    showWidgetError($("features-card").querySelector(".features-cols"), `Could not load features: ${err.message}`);
  }
}

/** Fill a <ul> with one <li> per item */
function fillList(list, items) {
  list.replaceChildren();
  (items || []).forEach((name) => list.appendChild(el("li", "", name)));
}

/* ---------- 4. Samples table ---------- */

async function loadSamples() {
  const body = $("samples-body");
  try {
    const data = await getJSON("/api/samples");
    renderSamples(body, data["Model Samples"] || []);
  } catch (err) {
    const row = el("tr");
    const cell = el("td", "widget-error", `Could not load samples: ${err.message}`);
    cell.colSpan = 8;
    row.appendChild(cell);
    body.replaceChildren(row);
  }
}

/** Render one table row per sample applicant */
function renderSamples(body, samples) {
  body.replaceChildren();

  samples.forEach((s) => {
    const tr = el("tr");

    // Dependents is stored as 0..3, where 3 means "3+"
    const dependents = s.Dependents === null || s.Dependents === undefined
      ? "—"
      : Number(s.Dependents) >= 3 ? "3+" : String(Number(s.Dependents));

    tr.appendChild(el("td", "num", dependents));
    tr.appendChild(el("td", "num", formatNumber(s.ApplicantIncome)));
    tr.appendChild(el("td", "num", formatNumber(s.CoapplicantIncome)));
    tr.appendChild(el("td", "num", formatNumber(s.LoanAmount)));
    tr.appendChild(el("td", "num", formatNumber(s.Credit_History)));
    tr.appendChild(el("td", "", s.Education ?? "—"));
    tr.appendChild(el("td", "", s.Self_Employed ?? "—"));

    // Loan_Status as a coloured badge
    const statusCell = el("td");
    const isApproved = s.Loan_Status === "Y";
    statusCell.appendChild(el("span", `badge ${isApproved ? "badge-y" : "badge-n"}`, s.Loan_Status ?? "—"));
    tr.appendChild(statusCell);

    body.appendChild(tr);
  });
}

/* ---------- 5. Model details ---------- */

async function loadInfo() {
  try {
    const data = await getJSON("/api/info");
    $("info-file").textContent = data["file name"] ?? "—";
    $("info-classes").textContent = Array.isArray(data.classes) ? data.classes.join(", ") : "—";
  } catch (err) {
    $("info-file").textContent = "Unavailable";
    $("info-classes").textContent = "Unavailable";
  }
}

/* ---------- 6. Prediction form ---------- */

/** Read a number input: empty -> null, otherwise a real number */
function readNumber(input) {
  const raw = input.value.trim();
  if (raw === "") return null;
  const num = Number(raw);
  return Number.isFinite(num) ? num : null;
}

/** Build the JSON payload with the exact keys and types the API expects */
function buildPayload(form) {
  const f = form.elements;
  return {
    ApplicantIncome: readNumber(f.ApplicantIncome),
    CoapplicantIncome: readNumber(f.CoapplicantIncome),
    LoanAmount: readNumber(f.LoanAmount),
    Dependents: Number(f.Dependents.value),          // 0..3
    Credit_History: Number(f.Credit_History.value),  // 1 or 0
    Education: f.Education.value,                    // "Graduate" | "Not Graduate"
    Self_Employed: f.Self_Employed.value,            // "Yes" | "No"
  };
}

async function handlePredict(event) {
  event.preventDefault();

  const form = event.currentTarget;
  const button = $("predict-btn");
  const result = $("predict-result");

  button.disabled = true;
  button.textContent = "Checking…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(buildPayload(form)),
    });
    const data = await res.json().catch(() => null);

    if (!res.ok) {
      // 400 / 503 return {"error": "..."}
      throw new Error((data && data.error) || `Request failed (${res.status})`);
    }
    renderPrediction(result, data);
  } catch (err) {
    const box = el("div", "result-box error", err.message);
    result.replaceChildren(box);
  } finally {
    button.disabled = false;
    button.textContent = "Check Loan Eligibility";
  }
}

/**
 * Show the approved / not approved box and the margin meter.
 * The meter spans -2..+2 with ticks at -1, 0, +1 (the margin edges and boundary).
 * Values beyond ±2 are pinned to the ends, but the exact value is always printed.
 */
function renderPrediction(container, data) {
  const approved = data.prediction === "Y";
  const margin = Number(data.margin);

  const box = el("div", `result-box ${approved ? "approved" : "rejected"}`);
  box.appendChild(el("div", "result-title", approved ? "Loan Approved" : "Loan Not Approved"));

  // Meter
  const RANGE = 2;
  const toPercentPos = (v) => ((Math.max(-RANGE, Math.min(RANGE, v)) + RANGE) / (2 * RANGE)) * 100;

  const meter = el("div", "meter");

  const head = el("div", "meter-head");
  head.appendChild(el("span", "", "Decision margin"));
  const val = el("strong", "", Number.isFinite(margin) ? margin.toFixed(3) : "—");
  head.appendChild(val);
  meter.appendChild(head);

  const track = el("div", "meter-track");
  [-1, 0, 1].forEach((t) => {
    const tick = el("div", `meter-tick${t === 0 ? " zero" : ""}`);
    tick.style.left = `${toPercentPos(t)}%`;
    track.appendChild(tick);
  });
  const pointer = el("div", "meter-pointer");
  track.appendChild(pointer);
  meter.appendChild(track);

  const labels = el("div", "meter-labels");
  [["−1", -1], ["0", 0], ["+1", 1]].forEach(([text, t]) => {
    const span = el("span", "", text);
    span.style.left = `${toPercentPos(t)}%`;
    labels.appendChild(span);
  });
  meter.appendChild(labels);

  box.appendChild(meter);
  container.replaceChildren(box);

  // Slide the pointer from the centre to its value
  if (Number.isFinite(margin)) {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { pointer.style.left = `${toPercentPos(margin)}%`; });
    });
  }
}
