/**
 * app.js — Student Performance Predictor frontend
 *
 * Responsibilities:
 *   1. Form validation (client-side)
 *   2. Submit → POST /api/predict → render results
 *   3. Feature importance tab switching
 *   4. EDA plots grid + lightbox
 *   5. Reset button
 */

"use strict";

/* ─────────────────────────────────────────────────────────────
   Constants
───────────────────────────────────────────────────────────── */
const FIELDS = [
  { id: "studytime",      min: 1,   max: 4,   type: "int",   label: "Study Time" },
  { id: "attendance_pct", min: 0,   max: 100, type: "float", label: "Attendance %" },
  { id: "G1",             min: 0,   max: 20,  type: "float", label: "Previous Marks (G1)" },
  { id: "failures",       min: 0,   max: 4,   type: "int",   label: "Previous Failures" },
  { id: "absences",       min: 0,   max: 93,  type: "int",   label: "Absences" },
];

const EDA_PLOTS = [
  { file: "plots/01_target_distribution.png",  caption: "Final Grade (G3) Distribution" },
  { file: "plots/02_passfail_pie.png",          caption: "Pass / Fail Distribution" },
  { file: "plots/03_feature_distributions.png", caption: "Feature Distributions" },
  { file: "plots/04_correlation_heatmap.png",   caption: "Correlation Heatmap" },
  { file: "plots/05_studytime_vs_g3.png",       caption: "Study Time vs Final Grade" },
  { file: "plots/06_attendance_vs_g3.png",      caption: "Attendance % vs Final Grade" },
  { file: "plots/07_failures_vs_g3.png",        caption: "Previous Failures vs Final Grade" },
];

/* ─────────────────────────────────────────────────────────────
   DOM references (grabbed once, reused)
───────────────────────────────────────────────────────────── */
const form         = document.getElementById("predict-form");
const submitBtn    = document.getElementById("submit-btn");
const btnText      = document.getElementById("btn-text");
const btnSpinner   = document.getElementById("btn-spinner");
const resetBtn     = document.getElementById("reset-btn");

const placeholder  = document.getElementById("placeholder");
const errorBanner  = document.getElementById("error-banner");
const errorList    = document.getElementById("error-list");
const resultCard   = document.getElementById("result-card");

const gradeNumber  = document.getElementById("grade-number");
const gradeBox     = document.getElementById("grade-box");
const verdictBadge = document.getElementById("verdict-badge");
const gradeLabelEl = document.getElementById("grade-label-text");
const confidenceEl = document.getElementById("confidence-text");

const probBarFill  = document.getElementById("prob-bar-fill");
const probMarker   = document.getElementById("prob-bar-marker");
const failPct      = document.getElementById("fail-pct");
const passPct      = document.getElementById("pass-pct");

const explanList   = document.getElementById("explanation-list");
const impReg       = document.getElementById("importance-reg");
const impClf       = document.getElementById("importance-clf");
const metricsGrid  = document.getElementById("metrics-grid");

const plotsGrid    = document.getElementById("plots-grid");
const lightbox     = document.getElementById("lightbox");
const lightboxImg  = document.getElementById("lightbox-img");
const lightboxClose= document.getElementById("lightbox-close");

/* ─────────────────────────────────────────────────────────────
   Validation helpers
───────────────────────────────────────────────────────────── */
function clearValidation() {
  FIELDS.forEach(({ id }) => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("invalid");
    const err = document.getElementById(`err-${id}`);
    if (err) err.textContent = "";
  });
}

function setFieldError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.classList.add("invalid");
  const err = document.getElementById(`err-${id}`);
  if (err) err.textContent = msg;
}

function validateForm() {
  clearValidation();
  let valid = true;

  FIELDS.forEach(({ id, min, max, type, label }) => {
    const el = document.getElementById(id);
    const raw = el ? el.value.trim() : "";

    if (raw === "") {
      setFieldError(id, `${label} is required.`);
      valid = false;
      return;
    }

    const num = type === "int" ? parseInt(raw, 10) : parseFloat(raw);
    if (isNaN(num)) {
      setFieldError(id, `${label} must be a number.`);
      valid = false;
      return;
    }

    if (num < min || num > max) {
      setFieldError(id, `${label} must be between ${min} and ${max}.`);
      valid = false;
    }
  });

  return valid;
}

function buildPayload() {
  const payload = {};
  FIELDS.forEach(({ id, type }) => {
    const raw = document.getElementById(id).value.trim();
    payload[id] = type === "int" ? parseInt(raw, 10) : parseFloat(raw);
  });
  return payload;
}

/* ─────────────────────────────────────────────────────────────
   UI state helpers
───────────────────────────────────────────────────────────── */
function setLoading(on) {
  submitBtn.disabled = on;
  btnText.textContent = on ? "Predicting…" : "Predict Performance";
  btnSpinner.classList.toggle("hidden", !on);
}

function showError(messages) {
  placeholder.classList.add("hidden");
  resultCard.classList.add("hidden");
  errorBanner.classList.remove("hidden");
  errorList.innerHTML = messages.map(m => `<li>${escHtml(m)}</li>`).join("");
}

function hideError() {
  errorBanner.classList.add("hidden");
  errorList.innerHTML = "";
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ─────────────────────────────────────────────────────────────
   Result rendering
───────────────────────────────────────────────────────────── */
function renderResult(data) {
  hideError();
  placeholder.classList.add("hidden");

  /* ── Grade box ── */
  gradeNumber.textContent = data.predicted_g3.toFixed(1);

  // Colour the grade box border by band
  gradeBox.style.borderColor =
    data.predicted_g3 >= 15 ? "#16a34a" :
    data.predicted_g3 >= 10 ? "#3b82f6" :
    data.predicted_g3 >= 5  ? "#d97706" : "#dc2626";

  /* ── Verdict badge ── */
  verdictBadge.textContent = data.pass_fail_label;
  verdictBadge.className   = "verdict-badge " + (data.pass_fail ? "pass" : "fail");
  gradeLabelEl.textContent = data.grade_label;
  confidenceEl.textContent = `Confidence: ${data.confidence_label}`;

  /* ── Probability bar ── */
  const passProbPct = Math.round(data.pass_probability * 100);
  const failProbPct = 100 - passProbPct;

  probBarFill.style.width   = `${passProbPct}%`;
  probMarker.style.left     = `${passProbPct}%`;
  failPct.textContent       = `${failProbPct}%`;
  passPct.textContent       = `${passProbPct}%`;
  failPct.style.color       = "var(--fail)";
  passPct.style.color       = "var(--pass)";

  /* ── Explanation ── */
  explanList.innerHTML = data.explanation
    .map(line => `<li>${escHtml(line)}</li>`)
    .join("");

  /* ── Feature importance ── */
  renderImportance(impReg, data.feature_importance_reg);
  renderImportance(impClf, data.feature_importance_clf);

  /* ── Model metrics ── */
  const rm = data.model_reg_metrics;
  const cm = data.model_clf_metrics;
  metricsGrid.innerHTML = [
    { name: "RMSE",     val: rm.rmse,     tip: "Lower is better",  sub: "Regression" },
    { name: "MAE",      val: rm.mae,      tip: "Lower is better",  sub: "Regression" },
    { name: "R²",       val: rm.r2,       tip: "Closer to 1 = better", sub: "Regression" },
    { name: "Accuracy", val: (cm.accuracy * 100).toFixed(1) + "%", tip: "", sub: "Classification" },
    { name: "F1 Score", val: cm.f1,       tip: "Precision & Recall balance", sub: "Classification" },
    { name: "AUC",      val: cm.auc,      tip: "Closer to 1 = better", sub: "Classification" },
  ].map(({ name, val, sub }) => `
    <div class="metric-tile">
      <div class="metric-name">${escHtml(name)}</div>
      <div class="metric-val">${typeof val === "number" ? val.toFixed(3) : val}</div>
      <div class="metric-model">${escHtml(sub)}</div>
    </div>
  `).join("");

  resultCard.classList.remove("hidden");

  /* Smooth scroll to results on mobile */
  if (window.innerWidth <= 860) {
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderImportance(container, items) {
  /* items = [{feature, importance}, ...] already sorted desc */
  const max = items[0]?.importance || 1;
  container.innerHTML = items.map((item, i) => {
    const pct = ((item.importance / max) * 100).toFixed(1);
    return `
      <div class="importance-row">
        <span class="importance-feat" title="${escHtml(item.feature)}">${escHtml(item.feature)}</span>
        <div class="importance-track">
          <div class="importance-fill ${i === 0 ? "top" : ""}"
               style="width:${pct}%"></div>
        </div>
        <span class="importance-val">${(item.importance * 100).toFixed(1)}%</span>
      </div>`;
  }).join("");
}

/* ─────────────────────────────────────────────────────────────
   Form submit handler
───────────────────────────────────────────────────────────── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!validateForm()) return;

  setLoading(true);
  hideError();

  try {
    const payload  = buildPayload();
    const response = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const json = await response.json();

    if (!json.success) {
      showError(json.errors || ["Unknown server error."]);
      return;
    }

    renderResult(json.data);

  } catch (err) {
    showError([`Network error: ${err.message}. Is the Flask server running?`]);
  } finally {
    setLoading(false);
  }
});

/* ─────────────────────────────────────────────────────────────
   Reset button
───────────────────────────────────────────────────────────── */
resetBtn.addEventListener("click", () => {
  form.reset();
  clearValidation();
  hideError();
  resultCard.classList.add("hidden");
  placeholder.classList.remove("hidden");
});

/* ─────────────────────────────────────────────────────────────
   Feature importance tabs
───────────────────────────────────────────────────────────── */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;

    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    document.getElementById("importance-reg").classList.toggle("hidden", tab !== "reg");
    document.getElementById("importance-clf").classList.toggle("hidden", tab !== "clf");
    document.getElementById("importance-reg").classList.toggle("tab-panel", true);
    document.getElementById("importance-clf").classList.toggle("tab-panel", true);

    if (tab === "reg") {
      document.getElementById("importance-reg").classList.remove("hidden");
    } else {
      document.getElementById("importance-clf").classList.remove("hidden");
    }
  });
});

/* ─────────────────────────────────────────────────────────────
   EDA plots grid
───────────────────────────────────────────────────────────── */
function buildPlotsGrid() {
  plotsGrid.innerHTML = EDA_PLOTS.map(({ file, caption }) => `
    <div class="plot-tile" data-src="/api/plots/${file}" tabindex="0"
         role="button" aria-label="Enlarge: ${escHtml(caption)}">
      <img src="/api/plots/${file}" alt="${escHtml(caption)}"
           loading="lazy" />
      <div class="plot-caption">${escHtml(caption)}</div>
    </div>
  `).join("");

  /* click to open lightbox */
  plotsGrid.querySelectorAll(".plot-tile").forEach(tile => {
    tile.addEventListener("click", () => openLightbox(tile.dataset.src));
    tile.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") openLightbox(tile.dataset.src);
    });
  });
}

function openLightbox(src) {
  lightboxImg.src = src;
  lightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.classList.add("hidden");
  lightboxImg.src = "";
  document.body.style.overflow = "";
}

lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", e => {
  if (e.target === lightbox) closeLightbox();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeLightbox();
});

/* ─────────────────────────────────────────────────────────────
   Init
───────────────────────────────────────────────────────────── */
buildPlotsGrid();
