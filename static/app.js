/**
 * NutriCal AI - Interactive Frontend Controller
 * Handles NLP parsing, ML inference requests, Chart.js visualizations,
 * and SQLite diary synchronization.
 */

// Global Chart instances
let nlpMacroChart = null;
let quickMacroChart = null;
let customMacroChart = null;
let featureImportanceChart = null;
let pcaClusterChart = null;

// Currently parsed meal data cache
let currentParsedMeal = null;

// Selected food in quick predictor
let selectedFoodItem = null;

// Initialize on document ready
document.addEventListener("DOMContentLoaded", () => {
  loadModelAnalytics();
  loadMealHistory();
  initDefaultQuickFood();
  updateCustomSliders();
});

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

  const targetTab = document.getElementById(tabId);
  if (targetTab) targetTab.classList.add("active");

  const btn = Array.from(document.querySelectorAll(".tab-btn")).find(b => 
    b.getAttribute("onclick") && b.getAttribute("onclick").includes(tabId)
  );
  if (btn) btn.classList.add("active");

  // Re-render chart sizes if needed
  if (tabId === "tab-analytics") {
    setTimeout(() => {
      if (featureImportanceChart) featureImportanceChart.resize();
      if (pcaClusterChart) pcaClusterChart.resize();
    }, 100);
  }
}

// Sample fillers for NLP input
function fillMealSample(text) {
  const input = document.getElementById("meal-nlp-input");
  input.value = text;
  parseMeal();
}

// NLP Meal Parser Handler
async function parseMeal() {
  const input = document.getElementById("meal-nlp-input");
  const text = input.value.trim();
  const statusMsg = document.getElementById("nlp-status-msg");
  const btn = document.getElementById("btn-parse-meal");

  if (!text) {
    showStatus(statusMsg, "Please enter what you ate first.", "error");
    return;
  }

  showStatus(statusMsg, "Parsing natural language meal & calculating calories...", "info");
  btn.disabled = true;

  try {
    const res = await fetch("/api/parse-meal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Failed to analyze meal.");
    }

    currentParsedMeal = data;
    renderNlpResults(data);
    showStatus(statusMsg, `Analyzed ${data.item_count} items successfully!`, "success");
  } catch (err) {
    showStatus(statusMsg, `Error: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

function renderNlpResults(data) {
  document.getElementById("nlp-empty-state").classList.add("hidden");
  const resultsContent = document.getElementById("nlp-results-content");
  resultsContent.classList.remove("hidden");

  // Total Calories & Meta
  document.getElementById("nlp-total-calories").innerText = data.total_calories;
  document.getElementById("nlp-item-count").innerText = data.item_count;

  // Macros
  document.getElementById("nlp-macro-p").innerText = `${data.total_protein}g`;
  document.getElementById("nlp-macro-c").innerText = `${data.total_carbs}g`;
  document.getElementById("nlp-macro-f").innerText = `${data.total_fat}g`;
  document.getElementById("nlp-macro-fib").innerText = `${data.total_fiber}g`;

  // Itemized List
  const itemsContainer = document.getElementById("nlp-items-list");
  itemsContainer.innerHTML = "";

  data.items.forEach(item => {
    const row = document.createElement("div");
    row.className = "item-row";
    row.innerHTML = `
      <div class="item-info">
        <strong>${item.matched_food}</strong>
        <div class="item-sub">${item.weight_g}g &bull; ${item.cooking_method} &bull; <em>"${item.original_phrase}"</em></div>
      </div>
      <div class="item-macros">
        <div class="item-cal">${item.calories} kcal</div>
        <div class="item-pcf">P: ${item.protein_g}g | C: ${item.carbs_g}g | F: ${item.fat_g}g</div>
      </div>
    `;
    itemsContainer.appendChild(row);
  });

  // Render Doughnut Chart
  renderDoughnutChart("nlpMacroChart", data.total_protein, data.total_carbs, data.total_fat);
}

// Chart.js Doughnut for Macro Breakdown
function renderDoughnutChart(canvasId, p, c, f) {
  const ctx = document.getElementById(canvasId).getContext("2d");

  const pCals = Math.round(p * 4.0);
  const cCals = Math.round(c * 4.0);
  const fCals = Math.round(f * 9.0);

  const chartData = {
    labels: ["Protein (kcal)", "Carbs (kcal)", "Fat (kcal)"],
    datasets: [{
      data: [pCals, cCals, fCals],
      backgroundColor: ["#06b6d4", "#f59e0b", "#f43f5e"],
      borderWidth: 0,
      hoverOffset: 6
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans", size: 11 }, boxWidth: 12 }
      }
    },
    cutout: "68%"
  };

  if (canvasId === "nlpMacroChart") {
    if (nlpMacroChart) nlpMacroChart.destroy();
    nlpMacroChart = new Chart(ctx, { type: "doughnut", data: chartData, options });
  } else if (canvasId === "quickMacroChart") {
    if (quickMacroChart) quickMacroChart.destroy();
    quickMacroChart = new Chart(ctx, { type: "doughnut", data: chartData, options });
  } else if (canvasId === "customMacroChart") {
    if (customMacroChart) customMacroChart.destroy();
    customMacroChart = new Chart(ctx, { type: "doughnut", data: chartData, options });
  }
}

// Save Meal to Diary
async function saveCurrentMealToDiary() {
  if (!currentParsedMeal) return;

  const mealNameInput = document.getElementById("diary-meal-name");
  const mealName = mealNameInput.value.trim() || "Daily Meal";

  try {
    const res = await fetch("/api/log-meal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        meal_name: mealName,
        total_calories: currentParsedMeal.total_calories,
        total_protein: currentParsedMeal.total_protein,
        total_carbs: currentParsedMeal.total_carbs,
        total_fat: currentParsedMeal.total_fat,
        items: currentParsedMeal.items
      })
    });

    const data = await res.json();
    if (data.success) {
      alert(`✅ "${mealName}" saved to your diary! Check the 'Meal Diary & Logs' tab.`);
      loadMealHistory();
    }
  } catch (err) {
    alert("Failed to save meal: " + err.message);
  }
}

// Quick Predictor Logic
async function initDefaultQuickFood() {
  try {
    const res = await fetch("/api/foods?q=Chicken%20Breast");
    const data = await res.json();
    if (data.success && data.foods.length > 0) {
      selectFood(data.foods[0]);
    }
  } catch (e) {
    console.warn("Could not load default food", e);
  }
}

let searchTimeout = null;
function handleFoodSearch(val) {
  clearTimeout(searchTimeout);
  const container = document.getElementById("food-suggestions");

  if (!val.trim()) {
    container.classList.add("hidden");
    return;
  }

  searchTimeout = setTimeout(async () => {
    try {
      const res = await fetch(`/api/foods?q=${encodeURIComponent(val)}`);
      const data = await res.json();
      if (data.success && data.foods.length > 0) {
        container.innerHTML = "";
        data.foods.forEach(f => {
          const item = document.createElement("div");
          item.className = "suggestion-item";
          item.innerHTML = `<span>${f.name}</span><span style="color:#64748b">${f.category}</span>`;
          item.onclick = () => selectFood(f);
          container.appendChild(item);
        });
        container.classList.remove("hidden");
      } else {
        container.classList.add("hidden");
      }
    } catch (e) {
      console.error(e);
    }
  }, 200);
}

function selectFood(food) {
  selectedFoodItem = food;
  document.getElementById("food-search-input").value = food.name;
  document.getElementById("food-suggestions").classList.add("hidden");

  // Set category dropdown
  const catSelect = document.getElementById("food-category-select");
  catSelect.value = food.category;

  // Set weight slider to standard item weight
  const weight = Math.round(food.standard_weight_g || 100);
  document.getElementById("portion-weight-slider").value = weight;
  document.getElementById("portion-weight-val").innerText = `${weight}g`;

  runQuickPrediction();
}

function updatePortionWeight(val) {
  document.getElementById("portion-weight-val").innerText = `${val}g`;
  runQuickPrediction();
}

function updateSelectedCategory() {
  runQuickPrediction();
}

async function runQuickPrediction() {
  const weight_g = parseFloat(document.getElementById("portion-weight-slider").value);
  const cooking_method = document.getElementById("cooking-method-select").value;
  const category = document.getElementById("food-category-select").value;
  const foodName = document.getElementById("food-search-input").value.trim() || "Selected Food";

  let p = 0, c = 0, f = 0, fib = 0, sug = 0, water = 70;

  if (selectedFoodItem && selectedFoodItem.name.toLowerCase() === foodName.toLowerCase()) {
    const ratio = weight_g / 100.0;
    p = selectedFoodItem.protein_g * ratio;
    c = selectedFoodItem.carbs_g * ratio;
    f = selectedFoodItem.fat_g * ratio;
    fib = selectedFoodItem.fiber_g * ratio;
    sug = selectedFoodItem.sugar_g * ratio;
    water = selectedFoodItem.water_pct;
  } else {
    // Fallback heuristic based on weight and category
    p = weight_g * 0.12;
    c = weight_g * 0.15;
    f = weight_g * 0.06;
    fib = weight_g * 0.02;
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        food_name: foodName,
        category: category,
        cooking_method: cooking_method,
        serving_weight_g: weight_g,
        protein_g: p,
        carbs_g: c,
        fat_g: f,
        fiber_g: fib,
        sugar_g: sug,
        water_pct: water
      })
    });

    const data = await res.json();
    if (data.success) {
      document.getElementById("quick-cal-pred").innerText = data.predicted_calories;
      document.getElementById("quick-ml-val").innerText = `${data.predicted_calories} kcal`;
      document.getElementById("quick-atwater-val").innerText = `${data.atwater_calories} kcal`;

      const diff = Math.abs(data.predicted_calories - data.atwater_calories);
      document.getElementById("quick-diff-val").innerText = `±${diff.toFixed(1)} kcal`;

      document.getElementById("quick-macro-p").innerText = `${data.macros.protein_g}g`;
      document.getElementById("quick-macro-c").innerText = `${data.macros.carbs_g}g`;
      document.getElementById("quick-macro-f").innerText = `${data.macros.fat_g}g`;
      document.getElementById("quick-macro-fib").innerText = `${data.macros.fiber_g}g`;

      renderDoughnutChart("quickMacroChart", data.macros.protein_g, data.macros.carbs_g, data.macros.fat_g);
    }
  } catch (e) {
    console.error("Quick prediction error:", e);
  }
}

// Custom Macro Lab Sliders
let customDebounce = null;
function updateCustomSliders() {
  const p = parseFloat(document.getElementById("custom-p-slider").value);
  const c = parseFloat(document.getElementById("custom-c-slider").value);
  const f = parseFloat(document.getElementById("custom-f-slider").value);
  const fib = parseFloat(document.getElementById("custom-fib-slider").value);
  const w = parseFloat(document.getElementById("custom-w-slider").value);

  document.getElementById("custom-p-val").innerText = `${p}g`;
  document.getElementById("custom-c-val").innerText = `${c}g`;
  document.getElementById("custom-f-val").innerText = `${f}g`;
  document.getElementById("custom-fib-val").innerText = `${fib}g`;
  document.getElementById("custom-w-val").innerText = `${w}%`;

  clearTimeout(customDebounce);
  customDebounce = setTimeout(async () => {
    const netCarbs = Math.max(0, c - fib);
    const atwater = (4.0 * p) + (4.0 * netCarbs) + (9.0 * f) + (2.0 * fib);
    const weightEstimate = Math.max(50, (p + c + f + fib) / (1 - (w / 100)));

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          food_name: "Lab Sample",
          category: "Prepared",
          cooking_method: "raw",
          serving_weight_g: weightEstimate,
          protein_g: p,
          carbs_g: c,
          fat_g: f,
          fiber_g: fib,
          sugar_g: c * 0.4,
          water_pct: w
        })
      });
      const data = await res.json();
      if (data.success) {
        document.getElementById("custom-cal-pred").innerText = data.predicted_calories;
        document.getElementById("custom-atwater-pred").innerText = `${data.atwater_calories} kcal`;
        const density = (data.predicted_calories / weightEstimate).toFixed(2);
        document.getElementById("custom-cal-density").innerText = `${density} kcal/g`;
        renderDoughnutChart("customMacroChart", p, c, f);
      }
    } catch (e) {
      console.error(e);
    }
  }, 100);
}

// Load Model Analytics & Unsupervised Clusters
async function loadModelAnalytics() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    if (!data.success) return;

    const meta = data.metadata || {};
    const metrics = meta.metrics || {};

    if (metrics.RidgeRegression) {
      document.getElementById("metric-ridge-r2").innerText = metrics.RidgeRegression.r2;
    }
    if (metrics.GradientBoosting) {
      document.getElementById("metric-gb-r2").innerText = metrics.GradientBoosting.r2;
    }
    if (metrics.RandomForest) {
      document.getElementById("metric-rf-r2").innerText = metrics.RandomForest.r2;
    }
    if (meta.sample_count) {
      document.getElementById("metric-samples").innerText = meta.sample_count.toLocaleString();
    }

    // Feature Importance Chart
    if (meta.feature_importances) {
      renderFeatureImportanceChart(meta.feature_importances);
    }

    // Unsupervised Clustering
    if (data.clustering) {
      renderClusterSummaries(data.clustering.cluster_summaries);
      renderPcaClusterChart(data.clustering.points);
    }
  } catch (err) {
    console.error("Failed to load analytics:", err);
  }
}

function renderFeatureImportanceChart(importances) {
  const ctx = document.getElementById("featureImportanceChart").getContext("2d");
  const labels = Object.keys(importances);
  const values = Object.values(importances);

  if (featureImportanceChart) featureImportanceChart.destroy();

  featureImportanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Relative Importance Weight",
        data: values,
        backgroundColor: [
          "#10b981", "#06b6d4", "#3b82f6", "#a855f7",
          "#f59e0b", "#ec4899", "#14b8a6", "#8b5cf6"
        ],
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#94a3b8" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#f8fafc", font: { size: 12 } }
        }
      }
    }
  });
}

function renderClusterSummaries(clusters) {
  const container = document.getElementById("cluster-summaries-list");
  container.innerHTML = "";

  clusters.forEach(c => {
    const item = document.createElement("div");
    item.className = "cluster-item";
    item.innerHTML = `
      <div class="cluster-title-row">
        <span class="cluster-name">${c.label}</span>
        <span class="cluster-count">${c.food_count} Foods</span>
      </div>
      <div class="cluster-stats">
        Avg Cal: <b>${c.avg_calories} kcal</b> | P: ${c.avg_protein}g | C: ${c.avg_carbs}g | F: ${c.avg_fat}g
      </div>
      <div class="cluster-samples">
        <em>e.g. ${c.sample_foods.join(", ")}</em>
      </div>
    `;
    container.appendChild(item);
  });
}

function renderPcaClusterChart(points) {
  const ctx = document.getElementById("pcaClusterChart").getContext("2d");

  const clusterColors = ["#06b6d4", "#f59e0b", "#f43f5e", "#10b981", "#a855f7"];

  const datasets = [0, 1, 2, 3, 4].map(cId => {
    const clusterPoints = points.filter(p => p.cluster_id === cId);
    const label = clusterPoints.length > 0 ? clusterPoints[0].cluster_label : `Cluster ${cId}`;
    return {
      label: label,
      data: clusterPoints.map(p => ({ x: p.x, y: p.y, name: p.name, calories: p.calories })),
      backgroundColor: clusterColors[cId % clusterColors.length],
      pointRadius: 6,
      pointHoverRadius: 9
    };
  });

  if (pcaClusterChart) pcaClusterChart.destroy();

  pcaClusterChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#cbd5e1", font: { size: 11 }, boxWidth: 10 }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const raw = ctx.raw;
              return `${raw.name}: ${raw.calories} kcal/100g (PCA: [${raw.x}, ${raw.y}])`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Principal Component 1 (Caloric & Fat Density)", color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b" }
        },
        y: {
          title: { display: true, text: "Principal Component 2 (Protein vs Carb Ratio)", color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b" }
        }
      }
    }
  });
}

// Load Meal History Table
async function loadMealHistory() {
  const tbody = document.getElementById("history-table-body");
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    if (!data.success || data.history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color:#64748b">No meals saved yet. Log a meal from the 'NLP Meal Logger' tab!</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    data.history.forEach(log => {
      const tr = document.createElement("tr");
      const dateStr = new Date(log.logged_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });
      const itemsCount = log.items ? log.items.length : 1;

      tr.innerHTML = `
        <td style="color:#94a3b8">${dateStr}</td>
        <td><strong>${log.meal_name}</strong></td>
        <td><span class="badge-accent">${itemsCount} items</span></td>
        <td style="color:#34d399; font-weight:700;">${log.total_calories} kcal</td>
        <td>${log.total_protein}g</td>
        <td>${log.total_carbs}g</td>
        <td>${log.total_fat}g</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color:#f43f5e">Error loading meal history: ${err.message}</td></tr>`;
  }
}

// Helper status display
function showStatus(elem, msg, type) {
  elem.innerText = msg;
  elem.style.display = "block";
  elem.style.marginTop = "12px";
  elem.style.fontSize = "13px";
  if (type === "error") {
    elem.style.color = "#f43f5e";
  } else if (type === "success") {
    elem.style.color = "#34d399";
  } else {
    elem.style.color = "#94a3b8";
  }
}
