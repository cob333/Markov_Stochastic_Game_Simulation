const state = {
  sessionId: null,
  autoplayTimer: null,
  snapshot: null,
  models: [],
};

const elements = {
  algoSelect: document.getElementById("algo-select"),
  seedInput: document.getElementById("seed-input"),
  speedInput: document.getElementById("speed-input"),
  speedLabel: document.getElementById("speed-label"),
  resetBtn: document.getElementById("reset-btn"),
  stepBtn: document.getElementById("step-btn"),
  autoplayBtn: document.getElementById("autoplay-btn"),
  serverStatus: document.getElementById("server-status"),
  sessionBadge: document.getElementById("session-badge"),
  stepCounter: document.getElementById("step-counter"),
  channelIndicator: document.getElementById("channel-indicator"),
  policyPath: document.getElementById("policy-path"),
  lastKTx: document.getElementById("last-k-tx"),
  nextCong: document.getElementById("next-cong"),
  metricsGrid: document.getElementById("metrics-grid"),
  deviceGrid: document.getElementById("device-grid"),
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

function sparklineSvg(values, color) {
  if (!values || values.length < 2) {
    return `<div class="empty-chart">运行会话后将显示曲线。</div>`;
  }

  const width = 640;
  const height = 230;
  const padding = 18;
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const valueSpan = maxVal - minVal || 1;
  const stepX = (width - padding * 2) / Math.max(1, values.length - 1);

  const points = values
    .map((value, index) => {
      const x = padding + index * stepX;
      const y =
        height - padding - ((value - minVal) / valueSpan) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="rgba(23,33,43,0.1)" stroke-width="2"></line>
      <polyline points="${points}" fill="none" stroke="${color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>
    </svg>
  `;
}

function renderMetrics(metrics) {
  const items = [
    ["吞吐量", formatNumber(metrics.throughput)],
    ["碰撞率", formatNumber(metrics.collision_rate)],
    ["单设备平均奖励", formatNumber(metrics.avg_reward_per_agent)],
    ["单设备平均能耗", formatNumber(metrics.avg_energy_per_agent)],
  ];

  elements.metricsGrid.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
}

function renderDevices(snapshot) {
  elements.deviceGrid.innerHTML = snapshot.agents
    .map(
      (agent) => `
        <article class="device-card">
          <header>
            <h3>设备 ${agent.id}</h3>
            <span class="observation">观测(信道=${agent.obs_channel}, 缓冲=${agent.obs_buffer})</span>
          </header>
          <div class="device-stats">
            <div class="device-stat">
              <span>缓存状态</span>
              <strong>${agent.buffer ? "有包" : "为空"}</strong>
            </div>
            <div class="device-stat">
              <span>最近动作</span>
              <strong>${agent.last_action_label}</strong>
            </div>
            <div class="device-stat">
              <span>最近奖励</span>
              <strong>${formatNumber(agent.last_reward)}</strong>
            </div>
            <div class="device-stat">
              <span>设备编号</span>
              <strong>#${agent.id}</strong>
            </div>
          </div>
          <div class="badges">
            <span class="badge ${agent.success ? "success" : ""}">
              ${agent.success ? "发送成功" : "未成功发送"}
            </span>
            <span class="badge ${agent.invalid ? "invalid" : ""}">
              ${agent.invalid ? "目标无效" : "目标有效"}
            </span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderEvents(snapshot) {
  elements.eventFeed.innerHTML = snapshot.recent_events
    .slice()
    .reverse()
    .map((event) => `<li>${event}</li>`)
    .join("");
}

function renderCharts(snapshot) {
  elements.throughputChart.innerHTML = sparklineSvg(
    snapshot.history.throughput,
    "#11716f"
  );
  elements.collisionChart.innerHTML = sparklineSvg(
    snapshot.history.collision_rate,
    "#d97706"
  );
}

function renderChannel(snapshot) {
  const isIdle = snapshot.channel.value === 0;
  elements.channelIndicator.textContent = snapshot.channel.label;
  elements.channelIndicator.classList.toggle("idle", isIdle);
  elements.channelIndicator.classList.toggle("congested", !isIdle);
  elements.stepCounter.textContent = `第 ${snapshot.step_index} 步`;
  elements.policyPath.textContent = snapshot.policy_path.split("/").slice(-1)[0];
  elements.lastKTx.textContent = snapshot.last_step ? snapshot.last_step.k_tx : "—";
  elements.nextCong.textContent = snapshot.last_step
    ? formatNumber(snapshot.last_step.p_congest_next)
    : "—";
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  elements.sessionBadge.textContent = `会话 ${snapshot.session_id.slice(0, 8)} · ${snapshot.algo_label}`;
  renderChannel(snapshot);
  renderMetrics(snapshot.metrics);
  renderDevices(snapshot);
  renderEvents(snapshot);
  renderCharts(snapshot);
  if (snapshot.terminated) {
    stopAutoplay();
    setStatus("当前回合已结束，请重置后重新开始。");
  } else {
    setStatus(`正在运行 ${snapshot.algo_label}，随机种子 ${snapshot.seed}`);
  }
}

async function loadModels() {
  try {
    const payload = await fetchJson("/api/models");
    state.models = payload.models;
    const options = payload.models
      .map((model) => {
        const disabled = model.available ? "" : "disabled";
        const suffix = model.available ? "" : ` (${model.reason})`;
        return `<option value="${model.id}" ${disabled}>${model.label}${suffix}</option>`;
      })
      .join("");
    elements.algoSelect.innerHTML = options;
    const firstAvailable = payload.models.find((model) => model.available);
    if (firstAvailable) {
      elements.algoSelect.value = firstAvailable.id;
      setStatus("模型加载完成。");
    } else {
      setStatus("当前没有可运行的模型。", true);
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function resetSession() {
  stopAutoplay();
  try {
    const payload = await fetchJson("/api/session/reset", {
      method: "POST",
      body: JSON.stringify({
        algo: elements.algoSelect.value,
        seed: Number(elements.seedInput.value || 0),
      }),
    });
    state.sessionId = payload.session_id;
    renderSnapshot(payload);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function stepSession() {
  if (!state.sessionId) {
    await resetSession();
    return;
  }
  try {
    const payload = await fetchJson("/api/session/step", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    renderSnapshot(payload);
  } catch (error) {
    stopAutoplay();
    setStatus(error.message, true);
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
    stepSession();
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
  elements.resetBtn.addEventListener("click", resetSession);
  elements.stepBtn.addEventListener("click", stepSession);
  elements.autoplayBtn.addEventListener("click", toggleAutoplay);
  elements.speedInput.addEventListener("input", () => {
    elements.speedLabel.textContent = `${elements.speedInput.value} 毫秒`;
    if (state.autoplayTimer) {
      startAutoplay();
    }
  });
}

async function bootstrap() {
  bindEvents();
  await loadModels();
  await resetSession();
}

bootstrap();
