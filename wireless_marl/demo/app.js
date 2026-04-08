const COMPARISON_MODELS = [
  { id: "value_iteration", label: "值迭代", color: "#11716f" },
  { id: "iql", label: "IQL", color: "#d97706" },
  { id: "qmix", label: "QMIX", color: "#c94f3d" },
  { id: "mappo", label: "MAPPO", color: "#3657b7" },
];

const TOPOLOGY_OPTIONS = {
  all_to_all: { label: "全连接" },
  star: { label: "星形" },
};

const state = {
  autoplayTimer: null,
  isStepping: false,
  models: [],
  sessionIds: {},
  sessionErrors: {},
  snapshots: {},
};

const elements = {
  seedInput: document.getElementById("seed-input"),
  topologySelect: document.getElementById("topology-select"),
  speedInput: document.getElementById("speed-input"),
  speedLabel: document.getElementById("speed-label"),
  resetBtn: document.getElementById("reset-btn"),
  stepBtn: document.getElementById("step-btn"),
  autoplayBtn: document.getElementById("autoplay-btn"),
  serverStatus: document.getElementById("server-status"),
  sessionBadge: document.getElementById("session-badge"),
  syncMode: document.getElementById("sync-mode"),
  syncStepCounter: document.getElementById("sync-step-counter"),
  seedDisplay: document.getElementById("seed-display"),
  activeModelCount: document.getElementById("active-model-count"),
  historyWindowLabel: document.getElementById("history-window-label"),
  topologyDisplay: document.getElementById("topology-display"),
  legendGrid: document.getElementById("legend-grid"),
  topologyName: document.getElementById("topology-name"),
  topologyType: document.getElementById("topology-type"),
  topologyDeviceCount: document.getElementById("topology-device-count"),
  topologyLinkCount: document.getElementById("topology-link-count"),
  topologySummary: document.getElementById("topology-summary"),
  topologyGraph: document.getElementById("topology-graph"),
  modelGrid: document.getElementById("model-grid"),
  eventFeed: document.getElementById("event-feed"),
  throughputChart: document.getElementById("throughput-chart"),
  collisionChart: document.getElementById("collision-chart"),
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `请求失败：${response.status}`);
  }
  return payload;
}

function setStatus(message, isError = false) {
  elements.serverStatus.textContent = message;
  elements.serverStatus.classList.toggle("error", isError);
}

function formatNumber(value, digits = 3) {
  return Number(value || 0).toFixed(digits);
}

function basename(path) {
  if (!path) {
    return "—";
  }
  return String(path).split("/").slice(-1)[0];
}

function comparisonModel(modelId) {
  return COMPARISON_MODELS.find((item) => item.id === modelId);
}

function modelMeta(modelId) {
  const preset = comparisonModel(modelId) || { id: modelId, label: modelId, color: "#243447" };
  const remote = state.models.find((item) => item.id === modelId);
  return {
    ...preset,
    label: remote?.label || preset.label,
    available: remote?.available ?? false,
    reason: remote?.reason || "",
    note: remote?.note || "",
    path: remote?.path || "",
    policyTopologyLabel: remote?.policy_topology_label || "",
  };
}

function currentTopology() {
  return elements.topologySelect.value || "all_to_all";
}

function topologyMeta(topologyId) {
  return TOPOLOGY_OPTIONS[topologyId] || { label: topologyId };
}

function activeSnapshots() {
  return COMPARISON_MODELS
    .map((item) => ({ ...item, snapshot: state.snapshots[item.id] || null }))
    .filter((item) => item.snapshot);
}

function firstSnapshot() {
  return activeSnapshots()[0]?.snapshot || null;
}

function topologySummary(type) {
  if (type === "all_to_all") {
    return "每个设备都可与其余设备直接连接。";
  }
  if (type === "ring") {
    return "每个设备仅与左右相邻设备连接。";
  }
  if (type === "star") {
    return "中心设备与其余设备连接，其余设备彼此不直连。";
  }
  return "按当前邻接矩阵进行连接。";
}

function fallbackTopology(snapshot, topologyType = currentTopology()) {
  const nAgents = snapshot?.agents?.length || snapshot?.buffers?.length || 4;
  let adjacency = Array.from({ length: nAgents }, () => Array.from({ length: nAgents }, () => 0));

  if (topologyType === "star") {
    adjacency = Array.from({ length: nAgents }, (_, row) =>
      Array.from({ length: nAgents }, (_, col) => {
        if (row === col) {
          return 0;
        }
        return row === 0 || col === 0 ? 1 : 0;
      })
    );
  } else {
    adjacency = Array.from({ length: nAgents }, (_, row) =>
      Array.from({ length: nAgents }, (_, col) => (row === col ? 0 : 1))
    );
  }

  return {
    type: topologyType,
    label: topologyMeta(topologyType).label,
    adjacency,
  };
}

function topologyGraphSvg(topology) {
  if (!topology || !topology.adjacency || topology.adjacency.length === 0) {
    return `<div class="empty-chart">加载会话后显示拓扑示意图。</div>`;
  }

  const adjacency = topology.adjacency;
  const nAgents = adjacency.length;
  const width = 440;
  const height = 260;
  const cx = width / 2;
  const cy = height / 2;
  const radius = 84;

  const positions = Array.from({ length: nAgents }, (_, index) => {
    const angle = (-Math.PI / 2) + (2 * Math.PI * index) / nAgents;
    return {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });

  const edges = [];
  for (let row = 0; row < nAgents; row += 1) {
    for (let col = row + 1; col < nAgents; col += 1) {
      if (adjacency[row][col] || adjacency[col][row]) {
        edges.push([row, col]);
      }
    }
  }

  const edgeSvg = edges
    .map(([from, to]) => {
      const start = positions[from];
      const end = positions[to];
      return `
        <line
          x1="${start.x}"
          y1="${start.y}"
          x2="${end.x}"
          y2="${end.y}"
          stroke="rgba(23,33,43,0.18)"
          stroke-width="4"
          stroke-linecap="round"
        ></line>
      `;
    })
    .join("");

  const nodeSvg = positions
    .map(
      (position, index) => `
        <g>
          <circle
            cx="${position.x}"
            cy="${position.y}"
            r="24"
            fill="rgba(17,113,111,0.14)"
            stroke="#11716f"
            stroke-width="3"
          ></circle>
          <text
            x="${position.x}"
            y="${position.y + 5}"
            text-anchor="middle"
            class="topology-node-label"
          >${index}</text>
        </g>
      `
    )
    .join("");

  return `
    <svg class="topology-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      ${edgeSvg}
      ${nodeSvg}
    </svg>
  `;
}

function multiLineChartSvg(seriesList, steps, yLabel) {
  const plottedSeries = seriesList.filter((item) => item.values && item.values.length > 0);
  const plottedSteps = steps && steps.length > 0
    ? steps
    : Array.from({ length: Math.max(...plottedSeries.map((item) => item.values.length), 0) }, (_, index) => index + 1);

  if (plottedSeries.length === 0 || plottedSteps.length < 2) {
    return `<div class="empty-chart">运行会话后将显示曲线。</div>`;
  }

  const width = 700;
  const height = 280;
  const margin = { top: 20, right: 24, bottom: 42, left: 56 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const stepX = chartWidth / Math.max(1, plottedSteps.length - 1);
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
  const xTickIndexes = [0, Math.floor((plottedSteps.length - 1) / 2), plottedSteps.length - 1]
    .filter((value, index, arr) => arr.indexOf(value) === index);

  const yGrid = yTicks
    .map((tick) => {
      const y = margin.top + chartHeight - tick * chartHeight;
      return `
        <line
          x1="${margin.left}"
          y1="${y}"
          x2="${width - margin.right}"
          y2="${y}"
          stroke="rgba(23,33,43,0.08)"
          stroke-width="1"
        ></line>
        <text
          x="${margin.left - 10}"
          y="${y + 4}"
          text-anchor="end"
          class="chart-tick"
        >${tick.toFixed(2)}</text>
      `;
    })
    .join("");

  const xGrid = xTickIndexes
    .map((tickIndex) => {
      const x = margin.left + tickIndex * stepX;
      const stepValue = Math.round(plottedSteps[tickIndex] || tickIndex + 1);
      let anchor = "middle";
      if (tickIndex === 0) {
        anchor = "start";
      } else if (tickIndex === plottedSteps.length - 1) {
        anchor = "end";
      }
      return `
        <line
          x1="${x}"
          y1="${height - margin.bottom}"
          x2="${x}"
          y2="${height - margin.bottom + 6}"
          stroke="rgba(23,33,43,0.16)"
          stroke-width="1.5"
        ></line>
        <text
          x="${x}"
          y="${height - margin.bottom + 22}"
          text-anchor="${anchor}"
          class="chart-tick"
        >${stepValue}</text>
      `;
    })
    .join("");

  const lineSvg = plottedSeries
    .map((series) => {
      const points = series.values
        .map((value, index) => {
          const clipped = Math.max(0, Math.min(1, value));
          const x = margin.left + index * stepX;
          const y = margin.top + chartHeight - clipped * chartHeight;
          return `${x},${y}`;
        })
        .join(" ");
      const lastIndex = series.values.length - 1;
      const lastValue = Math.max(0, Math.min(1, series.values[lastIndex]));
      const lastX = margin.left + lastIndex * stepX;
      const lastY = margin.top + chartHeight - lastValue * chartHeight;
      return `
        <polyline
          points="${points}"
          fill="none"
          stroke="${series.color}"
          stroke-width="4"
          stroke-linecap="round"
          stroke-linejoin="round"
        ></polyline>
        <circle cx="${lastX}" cy="${lastY}" r="4.5" fill="${series.color}"></circle>
      `;
    })
    .join("");

  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      ${yGrid}
      <line
        x1="${margin.left}"
        y1="${height - margin.bottom}"
        x2="${width - margin.right}"
        y2="${height - margin.bottom}"
        stroke="rgba(23,33,43,0.18)"
        stroke-width="2"
      ></line>
      <line
        x1="${margin.left}"
        y1="${margin.top}"
        x2="${margin.left}"
        y2="${height - margin.bottom}"
        stroke="rgba(23,33,43,0.18)"
        stroke-width="2"
      ></line>
      ${xGrid}
      ${lineSvg}
      <text x="${width / 2}" y="${height - 8}" text-anchor="middle" class="chart-axis-label">
        步数
      </text>
      <text
        x="18"
        y="${height / 2}"
        text-anchor="middle"
        transform="rotate(-90 18 ${height / 2})"
        class="chart-axis-label"
      >${yLabel}</text>
    </svg>
  `;
}

function historyWindowText(snapshot) {
  const windowSize = snapshot?.history_window || 20;
  return `最近 ${windowSize} 步滚动均值`;
}

function renderLegend() {
  elements.legendGrid.innerHTML = COMPARISON_MODELS
    .map((preset) => {
      const meta = modelMeta(preset.id);
      const statusClass = meta.available ? "ok" : "off";
      const statusText = meta.available
        ? (meta.note || "可运行")
        : (meta.reason || "不可用");
      return `
        <div class="legend-item ${statusClass}">
          <span class="legend-swatch" style="--legend-color:${preset.color}"></span>
          <div>
            <strong>${meta.label}</strong>
            <span>${statusText}</span>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderSyncSummary() {
  const snapshots = activeSnapshots();
  const currentStep = snapshots.length > 0
    ? Math.max(...snapshots.map((item) => item.snapshot.step_index))
    : 0;
  const primarySnapshot = firstSnapshot();
  const activeCount = snapshots.length;
  const topologyId = primarySnapshot?.topology?.type || currentTopology();
  const topologyLabel = topologyMeta(topologyId).label;

  elements.sessionBadge.textContent = `${topologyLabel}拓扑 · 同步会话 ${activeCount} / ${COMPARISON_MODELS.length}`;
  elements.syncMode.textContent = `四模型并行 · ${topologyLabel}拓扑`;
  elements.syncStepCounter.textContent = `第 ${currentStep} 步`;
  elements.seedDisplay.textContent = String(Number(elements.seedInput.value || 0));
  elements.activeModelCount.textContent = `${activeCount} / ${COMPARISON_MODELS.length}`;
  elements.historyWindowLabel.textContent = historyWindowText(primarySnapshot);
  elements.topologyDisplay.textContent = topologyLabel;
}

function renderTopology() {
  const snapshot = firstSnapshot();
  const topology = snapshot?.topology || fallbackTopology(snapshot, currentTopology());
  if (!topology) {
    elements.topologyName.textContent = "—";
    elements.topologyType.textContent = "—";
    elements.topologyDeviceCount.textContent = "—";
    elements.topologyLinkCount.textContent = "—";
    elements.topologySummary.textContent = "未获取到拓扑信息。";
    elements.topologyGraph.innerHTML = "加载会话后显示拓扑示意图。";
    return;
  }

  const adjacency = topology.adjacency || [];
  let edgeCount = 0;
  for (let row = 0; row < adjacency.length; row += 1) {
    for (let col = row + 1; col < adjacency.length; col += 1) {
      if (adjacency[row][col] || adjacency[col][row]) {
        edgeCount += 1;
      }
    }
  }

  elements.topologyName.textContent = topology.label;
  elements.topologyType.textContent = topology.label;
  elements.topologyDeviceCount.textContent = String(adjacency.length);
  elements.topologyLinkCount.textContent = String(edgeCount);
  elements.topologySummary.textContent = topologySummary(topology.type);
  elements.topologyGraph.innerHTML = topologyGraphSvg(topology);
}

function renderModelGrid() {
  elements.modelGrid.innerHTML = COMPARISON_MODELS
    .map((preset) => {
      const meta = modelMeta(preset.id);
      const snapshot = state.snapshots[preset.id];
      const sessionError = state.sessionErrors[preset.id];

      if (sessionError) {
        return `
          <article class="model-card unavailable" style="--model-color:${preset.color}">
            <div class="model-card-bar"></div>
            <header class="model-header">
              <div>
                <h3>${meta.label}</h3>
                <span class="model-subtitle">会话异常</span>
              </div>
              <span class="model-status-badge offline">错误</span>
            </header>
            <p class="model-message">${sessionError}</p>
          </article>
        `;
      }

      if (!meta.available) {
        return `
          <article class="model-card unavailable" style="--model-color:${preset.color}">
            <div class="model-card-bar"></div>
            <header class="model-header">
              <div>
                <h3>${meta.label}</h3>
                <span class="model-subtitle">模型不可用</span>
              </div>
              <span class="model-status-badge offline">离线</span>
            </header>
            <p class="model-message">${meta.reason || "缺少可运行模型。"}</p>
          </article>
        `;
      }

      if (!snapshot) {
        return `
          <article class="model-card" style="--model-color:${preset.color}">
            <div class="model-card-bar"></div>
            <header class="model-header">
              <div>
                <h3>${meta.label}</h3>
                <span class="model-subtitle">等待会话创建</span>
              </div>
              <span class="model-status-badge">待启动</span>
            </header>
            <p class="model-message">点击“重置会话”后将同步启动该模型。</p>
          </article>
        `;
      }

      const latestEvent = snapshot.recent_events.at(-1) || "尚无事件";
      const latestStep = snapshot.last_step;
      const statusText = snapshot.terminated ? "已结束" : "运行中";
      const statusClass = snapshot.terminated ? "offline" : "";
      const policyNote = snapshot.policy_topology !== snapshot.topology?.type
        ? `策略来源：${snapshot.policy_topology_label}`
        : basename(snapshot.policy_path);

      return `
        <article class="model-card" style="--model-color:${preset.color}">
          <div class="model-card-bar"></div>
          <header class="model-header">
            <div>
              <h3>${meta.label}</h3>
              <span class="model-subtitle">${policyNote}</span>
            </div>
            <span class="model-status-badge ${statusClass}">${statusText}</span>
          </header>
          <div class="model-kpis">
            <div class="model-kpi">
              <span>吞吐量</span>
              <strong>${formatNumber(snapshot.metrics.throughput)}</strong>
            </div>
            <div class="model-kpi">
              <span>碰撞率</span>
              <strong>${formatNumber(snapshot.metrics.collision_rate)}</strong>
            </div>
            <div class="model-kpi">
              <span>平均奖励</span>
              <strong>${formatNumber(snapshot.metrics.avg_reward_per_agent)}</strong>
            </div>
            <div class="model-kpi">
              <span>平均能耗</span>
              <strong>${formatNumber(snapshot.metrics.avg_energy_per_agent)}</strong>
            </div>
          </div>
          <div class="model-latest">
            <span>最近一步</span>
            <strong>
              ${
                latestStep
                  ? `发送设备数 ${latestStep.k_tx} · ${latestStep.collision ? "发生碰撞" : "无碰撞"}`
                  : "尚未步进"
              }
            </strong>
          </div>
          <div class="model-event">${latestEvent}</div>
        </article>
      `;
    })
    .join("");
}

function extractEventStep(eventText) {
  const match = /步数=(\d+)/.exec(eventText || "");
  return match ? Number(match[1]) : -1;
}

function renderEvents() {
  const mergedEvents = COMPARISON_MODELS
    .flatMap((preset) => {
      const snapshot = state.snapshots[preset.id];
      if (!snapshot) {
        return [];
      }
      return snapshot.recent_events.map((event) => ({
        label: modelMeta(preset.id).label,
        color: preset.color,
        text: event,
        step: extractEventStep(event),
      }));
    })
    .sort((left, right) => right.step - left.step)
    .slice(0, 12);

  if (mergedEvents.length === 0) {
    elements.eventFeed.innerHTML = "<li>启动同步会话后将显示模型事件。</li>";
    return;
  }

  elements.eventFeed.innerHTML = mergedEvents
    .map(
      (item) => `
        <li>
          <span class="event-model-chip" style="--legend-color:${item.color}">${item.label}</span>
          ${item.text}
        </li>
      `
    )
    .join("");
}

function chartSeries(metricKey) {
  return COMPARISON_MODELS
    .map((preset) => {
      const snapshot = state.snapshots[preset.id];
      if (!snapshot) {
        return null;
      }
      return {
        label: modelMeta(preset.id).label,
        color: preset.color,
        values: snapshot.history[metricKey] || [],
      };
    })
    .filter(Boolean);
}

function renderCharts() {
  const snapshot = firstSnapshot();
  const steps = snapshot?.history?.steps || [];
  elements.throughputChart.innerHTML = multiLineChartSvg(
    chartSeries("throughput_window"),
    steps,
    "吞吐量"
  );
  elements.collisionChart.innerHTML = multiLineChartSvg(
    chartSeries("collision_rate_window"),
    steps,
    "碰撞率"
  );
}

function renderDashboard() {
  renderLegend();
  renderSyncSummary();
  renderTopology();
  renderModelGrid();
  renderEvents();
  renderCharts();
}

function currentSeed() {
  return Number(elements.seedInput.value || 0);
}

async function loadModels() {
  try {
    const topology = currentTopology();
    const payload = await fetchJson(`/api/models?topology=${encodeURIComponent(topology)}`);
    state.models = payload.models;
    renderLegend();
    const availableCount = COMPARISON_MODELS.filter((item) => modelMeta(item.id).available).length;
    const fallbackCount = COMPARISON_MODELS.filter((item) => {
      const meta = modelMeta(item.id);
      return meta.available && Boolean(meta.note);
    }).length;
    if (availableCount > 0) {
      const suffix = fallbackCount > 0
        ? `，其中 ${fallbackCount} 个模型使用回退策略`
        : "";
      setStatus(`模型加载完成，可在${payload.topology_label}拓扑下同步运行 ${availableCount} 个模型${suffix}。`);
    } else {
      setStatus(`${payload.topology_label}拓扑下当前没有可运行的模型。`, true);
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function resetSessions() {
  stopAutoplay();
  state.sessionIds = {};
  state.sessionErrors = {};
  state.snapshots = {};
  renderDashboard();

  const seed = currentSeed();
  const topology = currentTopology();
  const topologyLabel = topologyMeta(topology).label;
  const runnable = COMPARISON_MODELS.filter((item) => modelMeta(item.id).available);
  if (runnable.length === 0) {
    setStatus(`${topologyLabel}拓扑下没有可启动的模型会话。`, true);
    return;
  }

  try {
    const results = await Promise.allSettled(
      runnable.map((item) =>
        fetchJson("/api/session/reset", {
          method: "POST",
          body: JSON.stringify({ algo: item.id, seed, topology }),
        })
      )
    );

    results.forEach((result, index) => {
      const modelId = runnable[index].id;
      if (result.status === "fulfilled") {
        state.sessionIds[modelId] = result.value.session_id;
        state.snapshots[modelId] = result.value;
      } else {
        state.sessionErrors[modelId] = result.reason?.message || "会话启动失败";
      }
    });

    renderDashboard();
    setStatus(`已在${topologyLabel}拓扑下同步启动 ${Object.keys(state.sessionIds).length} 个模型，随机种子 ${seed}。`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function stepSessions() {
  if (state.isStepping) {
    return;
  }
  if (Object.keys(state.sessionIds).length === 0) {
    await resetSessions();
    return;
  }

  state.isStepping = true;
  try {
    const targets = COMPARISON_MODELS.filter((item) => state.sessionIds[item.id]);
    const results = await Promise.allSettled(
      targets.map((item) =>
        fetchJson("/api/session/step", {
          method: "POST",
          body: JSON.stringify({ session_id: state.sessionIds[item.id] }),
        })
      )
    );

    results.forEach((result, index) => {
      const modelId = targets[index].id;
      if (result.status === "fulfilled") {
        delete state.sessionErrors[modelId];
        state.snapshots[modelId] = result.value;
      } else {
        state.sessionErrors[modelId] = result.reason?.message || "步进失败";
      }
    });

    renderDashboard();

    const snapshots = activeSnapshots().map((item) => item.snapshot);
    if (snapshots.length > 0 && snapshots.every((item) => item.terminated)) {
      stopAutoplay();
      setStatus("四模型同步回合已结束，请重置后重新开始。");
    } else {
      setStatus(`四模型在${topologyMeta(currentTopology()).label}拓扑下同步运行中，随机种子 ${currentSeed()}。`);
    }
  } catch (error) {
    stopAutoplay();
    setStatus(error.message, true);
  } finally {
    state.isStepping = false;
  }
}

function stopAutoplay() {
  if (state.autoplayTimer) {
    clearInterval(state.autoplayTimer);
    state.autoplayTimer = null;
  }
  elements.autoplayBtn.textContent = "开始自动播放";
  elements.autoplayBtn.classList.remove("active");
}

function startAutoplay() {
  stopAutoplay();
  const intervalMs = Number(elements.speedInput.value);
  state.autoplayTimer = window.setInterval(() => {
    stepSessions();
  }, intervalMs);
  elements.autoplayBtn.textContent = "停止自动播放";
  elements.autoplayBtn.classList.add("active");
}

function toggleAutoplay() {
  if (state.autoplayTimer) {
    stopAutoplay();
  } else {
    startAutoplay();
  }
}

function bindEvents() {
  elements.resetBtn.addEventListener("click", resetSessions);
  elements.stepBtn.addEventListener("click", stepSessions);
  elements.autoplayBtn.addEventListener("click", toggleAutoplay);
  elements.speedInput.addEventListener("input", () => {
    elements.speedLabel.textContent = `${elements.speedInput.value} 毫秒`;
    if (state.autoplayTimer) {
      startAutoplay();
    }
  });
  elements.seedInput.addEventListener("input", renderSyncSummary);
  elements.topologySelect.addEventListener("change", async () => {
    stopAutoplay();
    renderDashboard();
    await loadModels();
    await resetSessions();
  });
}

async function bootstrap() {
  bindEvents();
  renderDashboard();
  await loadModels();
  await resetSessions();
}

bootstrap();
