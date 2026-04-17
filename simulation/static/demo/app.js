const form = document.getElementById("launch-form");
const modelOptions = document.getElementById("model-options");
const quotaSelect = document.getElementById("quota-select");
const scenarioSelect = document.getElementById("scenario-select");
const modelInput = document.getElementById("model-input");
const launchStatus = document.getElementById("launch-status");
const sessionLabel = document.getElementById("session-label");
const activityConsole = document.getElementById("activity-console");
const replicaGrid = document.getElementById("replica-grid");
const koiReasoningPanel = document.getElementById("koi-reasoning-panel");
const koiReasoningText = document.getElementById("koi-reasoning-text");
const manualGpuType = document.getElementById("manual-gpu-type");
const manualTpSize = document.getElementById("manual-tp-size");
const manualPpSize = document.getElementById("manual-pp-size");
const manualReplicaCount = document.getElementById("manual-replica-count");
const addReplicaButton = document.getElementById("add-replica-button");
const killOldestButton = document.getElementById("kill-oldest-button");
const throttleButton = document.getElementById("throttle-button");
const themeToggle = document.getElementById("theme-toggle");
const koiThinkingBanner = document.getElementById("koi-thinking-banner");
const editQuotaButton = document.getElementById("edit-quota-button");
const quotaEditor = document.getElementById("quota-editor");
const quotaEditorBody = document.getElementById("quota-editor-body");
const quotaEditorLock = document.getElementById("quota-editor-lock");
const quotaEditorStatus = document.getElementById("quota-editor-status");
const quotaEditorSaveButton = document.getElementById("quota-editor-save");
const quotaEditorResetButton = document.getElementById("quota-editor-reset");

const quotaEditorState = {
  open: false,
  locked: false,
  presetSlug: null,
  workingValues: {},
};

let activeSource = null;
let catalog = null;
let currentSnapshot = null;
let manualControlSessionId = null;
const pageQuery = new URLSearchParams(window.location.search);
const previewMode = pageQuery.get("preview") === "1";
let previewReplicaCounter = 0;
const THEME_STORAGE_KEY = "koi_demo_theme";

function applyTheme(themeName) {
  const dark = themeName === "dark";
  document.body.classList.toggle("theme-dark", dark);
  if (themeToggle) {
    themeToggle.textContent = dark ? "Day Mode" : "Night Mode";
  }
}

function bootstrapTheme() {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if (savedTheme === "dark" || savedTheme === "light") {
    applyTheme(savedTheme);
    return;
  }
  applyTheme("light");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  return Number(value).toLocaleString();
}

function formatSeconds(value) {
  if (value === null || value === undefined) return "-";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
}

function formatClock(value) {
  if (!value) return "-";
  return new Date(value * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
}

function consoleToneForEntry(entry) {
  const key = `${entry.source}:${entry.kind}`;
  if (key.includes("trigger")) return "trigger";
  if (key.includes("tool")) return "tool";
  if (key.includes("scale") || key.includes("launch")) return "scale";
  return "neutral";
}

function titleCase(value) {
  return String(value || "")
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function phaseTone(phase) {
  const normalized = String(phase || "").toLowerCase();
  if (["running", "completed"].includes(normalized)) return "good";
  if (["launching", "provisioned", "bootstrapping", "searching_capacity", "waiting_model_ready"].includes(normalized)) return "pending";
  if (["failed", "dead", "killed"].includes(normalized)) return "bad";
  return "neutral";
}

function renderCatalog(data) {
  catalog = data;

  modelOptions.innerHTML = "";
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_name;
    modelOptions.appendChild(option);
  });
  if (!modelInput.value && data.models[0]) {
    modelInput.value = data.models[0].model_name;
  }

  quotaSelect.innerHTML = data.quota_presets
    .map((preset) => `<option value="${preset.slug}">${preset.title}</option>`)
    .join("");

  scenarioSelect.innerHTML = data.scenarios
    .map((scenario) => `<option value="${scenario.slug}">${scenario.title}</option>`)
    .join("");
  if (!scenarioSelect.value && data.scenarios[0]) {
    scenarioSelect.value = data.scenarios[0].slug;
  }

  updateQuotaLockState({
    locked: Boolean(data.quota_locked),
    activeSessions: Number(data.active_sessions || 0),
  });

  renderSelectedQuota();
  renderSelectedScenario();
  syncManualScaleControls();
}

function findQuotaPreset(slug) {
  if (!catalog) return null;
  return (catalog.quota_presets || []).find((preset) => preset.slug === slug) || null;
}

function applyPresetOverridesInPlace(preset) {
  if (!preset) return preset;
  const overrides = preset.overrides || {};
  if (!Object.keys(overrides).length) return preset;
  const familyRegionMarket = (row) =>
    `${String(row.family || "").toUpperCase()}|${String(row.region || "")}|${String(row.market || "")}`;
  preset.quotas = (preset.quotas || []).map((row) => {
    const key = familyRegionMarket(row);
    if (key in overrides) {
      return {...row, baseline_vcpus: Number(overrides[key] || 0)};
    }
    return row;
  });
  return preset;
}

function updateQuotaLockState({locked, activeSessions}) {
  quotaEditorState.locked = Boolean(locked);
  if (editQuotaButton) {
    editQuotaButton.title = locked
      ? `Locked — ${activeSessions || 1} active session`
      : "Edit quota ceilings (persists across restarts)";
  }
  if (quotaEditor && quotaEditorState.open) {
    renderQuotaEditorLock(activeSessions);
  }
}

function renderQuotaEditorLock(activeSessions) {
  if (!quotaEditorLock) return;
  if (quotaEditorState.locked) {
    quotaEditorLock.hidden = false;
    quotaEditorLock.textContent = `Quota editing is locked — ${activeSessions || 1} live session${(activeSessions || 1) === 1 ? "" : "s"} running. Let the current run finish before saving changes.`;
  } else {
    quotaEditorLock.hidden = true;
  }
  if (quotaEditorSaveButton) {
    quotaEditorSaveButton.disabled = quotaEditorState.locked;
  }
  if (quotaEditorResetButton) {
    quotaEditorResetButton.disabled = quotaEditorState.locked;
  }
}

const QUOTA_GPU_ORDER = ["A100", "L40", "L4", "A10G"];

function normalizeQuotaGpuName(rawGpu) {
  const gpu = String(rawGpu || "").toUpperCase();
  if (!gpu) return null;
  if (gpu.includes("H100")) return null;
  if (gpu.includes("A100")) return "A100";
  if (gpu.startsWith("L40") || gpu.includes("L40S")) return "L40";
  if (gpu.includes("L4")) return "L4";
  if (gpu.includes("A10G")) return "A10G";
  return null;
}

function liveQuotaViewForSnapshot(snapshot) {
  const liveResources = (((snapshot || {}).koi || {}).live || {}).resources || {};
  if (Array.isArray(liveResources.instances) && Array.isArray(liveResources.quotas)
    && (liveResources.instances.length || liveResources.quotas.length)) {
    return liveResources;
  }
  return null;
}

function baseQuotaViewForSnapshot(snapshot) {
  if (snapshot && snapshot.resource_map) {
    return snapshot.resource_map;
  }
  return catalog.quota_presets.find((item) => item.slug === quotaSelect.value) || null;
}

function renderSelectedQuota(snapshot = null) {
  if (!catalog) return;
  const quota = baseQuotaViewForSnapshot(snapshot);
  const liveQuota = liveQuotaViewForSnapshot(snapshot);
  const quotaRoot = document.getElementById("quota-details");
  if (!quotaRoot) return;
  if (!quota) {
    quotaRoot.innerHTML = "";
    return;
  }

  const rows = new Map(
    QUOTA_GPU_ORDER.map((gpu) => [gpu, {
      gpu,
      total: 0,
      used: 0,
      displayRegion: null,
      displayMarket: null,
      displayScore: -1,
    }]),
  );

  const familyToInstance = new Map();
  for (const instance of quota.instances || []) {
    const family = String(instance.quota_family || "").toUpperCase();
    if (family && !familyToInstance.has(family)) {
      familyToInstance.set(family, instance);
    }
  }

  const liveQuotaByKey = new Map();
  for (const quotaEntry of (liveQuota || {}).quotas || []) {
    const family = String(quotaEntry.family || "").toUpperCase();
    const region = String(quotaEntry.region || "");
    const market = String(quotaEntry.market || "");
    if (!family || !region || !market) continue;
    liveQuotaByKey.set(`${family}|${region}|${market}`, quotaEntry);
  }

  for (const quotaEntry of quota.quotas || []) {
    const family = String(quotaEntry.family || "").toUpperCase();
    const instance = familyToInstance.get(family);
    if (!instance) continue;

    const gpu = normalizeQuotaGpuName(instance.gpu_type);
    if (!gpu || !rows.has(gpu)) continue;

    const baselineVcpus = Number(quotaEntry.baseline_vcpus || 0);
    const liveKey = `${family}|${String(quotaEntry.region || "")}|${String(quotaEntry.market || "")}`;
    const liveQuotaEntry = liveQuotaByKey.get(liveKey);
    const usedVcpus = Number((liveQuotaEntry || quotaEntry).used_vcpus || 0);
    const instanceVcpus = Number(instance.vcpus || 0);
    const gpusPerInstance = Number(instance.gpus_per_instance || 0);

    if (baselineVcpus > 0 && instanceVcpus > 0 && gpusPerInstance > 0) {
      const totalGpu = Math.max(0, Math.round((baselineVcpus / instanceVcpus) * gpusPerInstance));
      const usedGpu = Math.max(0, Math.round((usedVcpus / instanceVcpus) * gpusPerInstance));
      const row = rows.get(gpu);
      row.total += totalGpu;
      row.used += Math.min(usedGpu, totalGpu);

      // Prefer the live in-use location, otherwise the largest quota row.
      const displayScore = (usedGpu > 0 ? 1_000_000 : 0) + totalGpu;
      if (displayScore > row.displayScore) {
        row.displayScore = displayScore;
        row.displayRegion = quotaEntry.region ? String(quotaEntry.region) : row.displayRegion;
        row.displayMarket = quotaEntry.market ? String(quotaEntry.market).replaceAll("_", " ") : row.displayMarket;
      }
    }
  }

  for (const instance of quota.instances || []) {
    const gpu = normalizeQuotaGpuName(instance.gpu_type);
    if (!gpu || !rows.has(gpu)) continue;
    const row = rows.get(gpu);
    if (row.total === 0) {
      row.total = Math.max(row.total, Number(instance.gpus_per_instance || 0));
    }
  }

  const renderedRows = QUOTA_GPU_ORDER
    .map((gpu) => rows.get(gpu))
    .filter((row) => row && (row.total > 0 || row.used > 0))
    .map((row) => {
      const pct = row.total > 0 ? Math.max(0, Math.min(100, (row.used / row.total) * 100)) : 0;
      const region = row.displayRegion || "us-east-1";
      const market = row.displayMarket || "on demand";
      return `
        <div class="quota-item">
          <div class="quota-item-header">
            <span class="quota-item-title">${escapeHtml(row.gpu)}</span>
            <span class="quota-item-count">${row.used}/${row.total}</span>
          </div>
          <div class="quota-item-sub">${escapeHtml(region)} · ${escapeHtml(market)}</div>
          <div class="quota-bar-track">
            <div class="quota-bar-fill" style="width:${pct}%"></div>
          </div>
        </div>
      `;
    });

  quotaRoot.innerHTML = renderedRows.length
    ? renderedRows.join("")
    : `<div class="quota-empty">No AWS GPU quota rows for A100, L40, L4, or A10G.</div>`;
}

function renderSelectedScenario() {
  if (!catalog) return;
  const scenario = catalog.scenarios.find((item) => item.slug === scenarioSelect.value);
  const root = document.getElementById("scenario-details");
  if (!root) return;
  if (!scenario) {
    root.textContent = "";
    return;
  }
  const events = scenario.events.length
    ? scenario.events.map((event) => `${event.label} @ ${event.at_seconds}s`).join(" · ")
    : "No timed events";
  root.innerHTML = `
    <p>${escapeHtml(scenario.description)}</p>
    <p class="quota-meta">Initial replicas: ${scenario.initial_replicas} · Launch multiplier: ${scenario.launch_timing_multiplier}x</p>
    <p class="quota-meta">${escapeHtml(events)}</p>
  `;
}

function getManualResourceMap(snapshot = null) {
  if (snapshot && snapshot.resource_map) {
    return snapshot.resource_map;
  }
  const preset = (catalog?.quota_presets || []).find((item) => item.slug === quotaSelect.value);
  return preset || {instances: [], quotas: []};
}

function getManualGpuOptions(snapshot = null) {
  const resourceMap = getManualResourceMap(snapshot);
  const seen = new Set();
  return (resourceMap.instances || []).filter((instance) => {
    const gpuType = String(instance.gpu_type || "");
    if (!gpuType || seen.has(gpuType)) return false;
    seen.add(gpuType);
    return true;
  });
}

function getManualInstanceForGpu(snapshot, gpuType) {
  return getManualGpuOptions(snapshot).find((instance) => instance.gpu_type === gpuType) || null;
}

function getManualQuotaForGpu(snapshot, gpuType) {
  const instance = getManualInstanceForGpu(snapshot, gpuType);
  if (!instance) return null;
  const family = String(instance.quota_family || "").toUpperCase();
  return (getManualResourceMap(snapshot).quotas || []).find((quota) => String(quota.family || "").toUpperCase() === family) || null;
}

function getDefaultManualConfig(snapshot = null) {
  const launchConfig = (snapshot || {}).launch_config || {};
  const fallback = getCurrentConfig(snapshot || currentSnapshot || {
    launch_preview: {preferred_gpu: getManualGpuOptions(snapshot)[0]?.gpu_type || "L40S", tp: 4, pp: 1},
    koi: {},
  });
  return {
    gpu_type: launchConfig.gpu_type || fallback.gpu_type,
    tp: Number(launchConfig.tp || fallback.tp || 4),
    pp: Number(launchConfig.pp || fallback.pp || 1),
  };
}

function syncManualScaleControls(snapshot = null) {
  if (!manualGpuType || !manualTpSize || !manualPpSize || !manualReplicaCount) return;
  const options = getManualGpuOptions(snapshot);
  const optionMarkup = options.length
    ? options.map((instance) => `<option value="${escapeHtml(instance.gpu_type)}">${escapeHtml(instance.gpu_type)}</option>`).join("")
    : '<option value="">No GPU options</option>';
  const previousGpu = manualGpuType.value;
  const sessionId = (snapshot || {}).session_id || "catalog";
  const sessionChanged = sessionId !== manualControlSessionId;
  const defaults = getDefaultManualConfig(snapshot);

  manualGpuType.innerHTML = optionMarkup;
  const validGpuValues = options.map((instance) => instance.gpu_type);
  const nextGpu = (!sessionChanged && validGpuValues.includes(previousGpu))
    ? previousGpu
    : (validGpuValues.includes(defaults.gpu_type) ? defaults.gpu_type : (validGpuValues[0] || ""));
  manualGpuType.value = nextGpu;

  if (sessionChanged) {
    manualTpSize.value = String(Math.max(1, defaults.tp));
    manualPpSize.value = String(Math.max(1, defaults.pp));
    manualReplicaCount.value = String(Math.max(1, Number(manualReplicaCount.value || 1)));
    manualControlSessionId = sessionId;
  }
}

function getManualScaleConfig(snapshot = null) {
  const activeSnapshot = snapshot || currentSnapshot;
  const defaults = getDefaultManualConfig(activeSnapshot);
  return {
    gpu_type: manualGpuType?.value || defaults.gpu_type,
    tp: Math.max(1, Number(manualTpSize?.value || defaults.tp || 4)),
    pp: Math.max(1, Number(manualPpSize?.value || defaults.pp || 1)),
    count: Math.max(1, Number(manualReplicaCount?.value || 1)),
  };
}

function renderSession(snapshot) {
  currentSnapshot = snapshot;
  previewReplicaCounter = Math.max(
    previewReplicaCounter,
    ...((((snapshot.runtime || {}).replicas) || []).map((replica) => {
      const match = String(replica.replica_id || "").match(/-r(\d+)$/);
      return match ? Number(match[1]) + 1 : 0;
    })),
  );
  const runtime = snapshot.runtime;
  sessionLabel.textContent = `${snapshot.session_id} · ${snapshot.model.model_name}`;
  syncManualScaleControls(snapshot);

  const koiBusy = isKoiBusy(snapshot);
  if (koiThinkingBanner) {
    koiThinkingBanner.hidden = !koiBusy;
  }

  document.getElementById("runtime-status").textContent = runtime.status;
  const launchPhase = document.getElementById("launch-phase");
  if (launchPhase) {
    launchPhase.textContent = runtime.launch_phase;
  }
  document.getElementById("aggregate-tps").textContent = formatNumber(runtime.aggregate_tps);
  document.getElementById("eta-seconds").textContent = formatSeconds(runtime.eta_seconds);
  document.getElementById("active-replicas").textContent = formatNumber(runtime.active_replicas);
  document.getElementById("progress-percent").textContent = `${runtime.progress_pct}%`;
  document.getElementById("slo-headroom").textContent = runtime.slo_headroom_pct === null ? "-" : `${runtime.slo_headroom_pct}%`;
  document.getElementById("token-progress").textContent = `${formatNumber(runtime.tokens_completed)} / ${formatNumber(runtime.tokens_total)} tokens`;
  document.getElementById("progress-fill").style.width = `${runtime.progress_pct}%`;

  const modelDetails = document.getElementById("model-details");
  const modelRows = [
    ["Resolution", snapshot.model.source],
    ["Params", `${snapshot.model.num_params_billions}B`],
    ["Active Params", `${snapshot.model.active_params_billions}B`],
    ["Model Size", `${snapshot.model.model_size_gb.toFixed(1)} GB`],
    ["Family", snapshot.model.architecture_family],
    ["MoE", snapshot.model.is_moe ? "yes" : "no"],
  ];
  modelDetails.innerHTML = modelRows.map(([key, value]) => `
    <div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>
  `).join("");

  const koiRoot = document.getElementById("koi-details");
  if (snapshot.koi && snapshot.koi.decision_status === "pending") {
    koiRoot.innerHTML = `
      <p><strong>Koi is deciding.</strong> The session already exists and the launch will start as soon as the decision lands.</p>
      <p class="quota-meta">Launch preview uses the current fallback profile until Koi picks the real config.</p>
    `;
  } else if (snapshot.koi && snapshot.koi.decision) {
    const decision = snapshot.koi.decision;
    const cfg = decision.config || {};
    const liveJobs = (((snapshot.koi || {}).live || {}).jobs || {}).jobs || [];
    const liveResources = (((snapshot.koi || {}).live || {}).resources) || {};
    const liveSummary = snapshot.koi.live
      ? `
        <p class="quota-meta">Live Koi jobs: ${liveJobs.length} · Pending launch reservations: ${liveResources.pending_count ?? 0}</p>
        ${liveJobs.length ? `<p class="quota-meta">Tracked job IDs: ${escapeHtml(liveJobs.map((job) => job.job_id).join(", "))}</p>` : ""}
      `
      : "";
    const syncStatus = snapshot.koi.sync
      ? `<p class="quota-meta">Sync: ${escapeHtml(snapshot.koi.sync.status)}</p>`
      : "";
    const syncError = snapshot.koi.sync_error
      ? `<p class="quota-meta">Sync error: ${escapeHtml(snapshot.koi.sync_error)}</p>`
      : "";
    koiRoot.innerHTML = `
      <p><strong>Decision:</strong> ${escapeHtml(cfg.gpu_type || "-")} · TP ${cfg.tp || "-"} · PP ${cfg.pp || "-"}</p>
      <p class="quota-meta">Predicted TPS: ${formatNumber(decision.predicted_tps)} · Confidence: ${decision.confidence || "-"}</p>
      <p class="quota-meta">Decision ID: ${escapeHtml(decision._decision_id || "-")}</p>
      ${syncStatus}
      ${syncError}
      ${liveSummary}
    `;
  } else if (snapshot.koi && snapshot.koi.error) {
    koiRoot.innerHTML = `<p>Koi unavailable: ${snapshot.koi.error}</p>`;
  } else if (snapshot.koi && snapshot.koi.decision_status === "fallback") {
    koiRoot.innerHTML = `<p><strong>Koi decision failed.</strong> The simulator is using fallback launch defaults so the run can continue.</p>`;
  } else {
    koiRoot.innerHTML = "<p>Live Koi decision not attached. Using demo runtime defaults.</p>";
  }

  renderSelectedQuota(snapshot);
  renderReplicaFleet(snapshot);
  renderActivityConsole(snapshot);
  renderKoiReasoningPanel(snapshot);
}

function renderKoiEventDetails(event) {
  const parts = [];
  if (event.job_id) parts.push(`job ${event.job_id}`);
  if (event.group_id) parts.push(`group ${event.group_id}`);
  if (event.tool) parts.push(`tool ${event.tool}`);
  if (event.trigger_type) parts.push(`trigger ${event.trigger_type}`);
  if (event.phase) parts.push(`phase ${event.phase}`);
  if (event.response) parts.push(event.response);
  return parts.join(" · ");
}

function isKoiBusy(snapshot) {
  const koi = (snapshot || {}).koi || {};
  if (koi.decision_status === "pending") {
    return true;
  }
  const events = koi.events || [];
  if (!events.length) {
    return false;
  }
  const lastEvent = String(events[events.length - 1].event || "");
  return ["agent_deciding", "tool_call", "trigger_handling"].includes(lastEvent);
}

function renderKoiReasoningPanel(snapshot) {
  if (!koiReasoningText) {
    return;
  }

  const events = ((((snapshot || {}).koi || {}).events) || []).slice(-80);
  const decision = (((snapshot || {}).koi || {}).decision) || null;
  const busy = isKoiBusy(snapshot);
  const latestActivity = events.length
    ? describeKoiEvent(events[events.length - 1]).title
    : "";

  if (decision && decision.reasoning) {
    koiReasoningText.textContent = decision.reasoning;
    koiReasoningText.classList.remove("is-empty");
  } else if (busy) {
    koiReasoningText.textContent = latestActivity
      ? `Koi is actively working. Latest step: ${latestActivity}.`
      : "Koi is actively deciding.";
    koiReasoningText.classList.remove("is-empty");
  } else {
    koiReasoningText.textContent = "No Koi reasoning yet. Launch a session to inspect Koi's decision narrative here.";
    koiReasoningText.classList.add("is-empty");
  }

  if (koiReasoningPanel) {
    koiReasoningPanel.dataset.state = busy
      ? "thinking"
      : (decision ? "ready" : "idle");
  }
}

function describeRuntimeEvent(event) {
  const label = event.label || titleCase(event.action || "Event");
  return {
    title: label,
    detail: event.description || "",
  };
}

function describeKoiEvent(event) {
  const eventName = String(event.event || "event");
  if (eventName === "agent_deciding") {
    return {
      title: "Koi started deciding",
      detail: `Evaluating placement for ${event.job_id || "the job"}.`,
    };
  }
  if (eventName === "tool_call") {
    const toolName = event.tool || event.label || "tool";
    const detail = event.call_number
      ? `Tool call #${event.call_number}${toolName ? ` (${toolName})` : ""}.`
      : "Inspecting the workload and cluster state.";
    return {
      title: `Koi called ${event.label || "decide"}`,
      detail,
    };
  }
  if (eventName === "agent_decided") {
    const elapsed = event.elapsed_s ? `${event.elapsed_s}s` : "unknown time";
    const toolCalls = event.tool_calls ?? event.tool_calls_made;
    return {
      title: "Koi produced a decision",
      detail: toolCalls ? `Finished in ${elapsed} after ${toolCalls} tool calls.` : `Finished in ${elapsed}.`,
    };
  }
  if (eventName === "trigger_handling") {
    return {
      title: `Koi handling ${titleCase(event.trigger_type)}`,
      detail: `Reacting to ${event.job_id || "the active job"}.`,
    };
  }
  if (eventName === "trigger_response") {
    return {
      title: "Koi chose a recovery action",
      detail: event.response || "A response was generated for the current trigger.",
    };
  }
  if (eventName === "job_launching") {
    return {
      title: `${event.job_id || "Replica"} provisioned`,
      detail: "Waiting for model_ready before tracking throughput.",
    };
  }
  if (eventName === "job_launch_heartbeat") {
    return {
      title: `${event.job_id || "Replica"} still launching`,
      detail: event.message || `Phase: ${titleCase(event.phase)}`,
    };
  }
  if (eventName === "job_started") {
    return {
      title: `${event.job_id || "Replica"} is running`,
      detail: `${event.gpu_type || "GPU"} · TP ${event.tp || "-"} · PP ${event.pp || "-"}`,
    };
  }
  if (eventName === "job_launch_failed") {
    return {
      title: "Launch attempt failed",
      detail: event.error || "Orca reported that all candidate launch attempts failed.",
    };
  }
  if (eventName === "job_complete") {
    return {
      title: `${event.job_id || "Job"} completed`,
      detail: "Final outcome recorded in Koi.",
    };
  }
  if (eventName === "job_replica_failed") {
    return {
      title: `${event.job_id || "Replica"} failed`,
      detail: event.reason || "Koi was notified that a replica died mid-run.",
    };
  }

  return {
    title: titleCase(eventName),
    detail: renderKoiEventDetails(event),
  };
}

function renderReplicaFleet(snapshot) {
  if (!replicaGrid) return;
  const replicas = (snapshot.runtime && snapshot.runtime.replicas) || [];
  if (!replicas.length) {
    replicaGrid.innerHTML = `
      <div class="replica-table-empty">
        <div class="infra-empty-title">No live replicas yet</div>
        <div class="infra-empty-body">Launch a session to see the worker fleet and control buttons here.</div>
      </div>
    `;
    return;
  }

  replicaGrid.innerHTML = `
    <table class="replica-tbl">
      <thead>
        <tr>
          <th>Replica</th>
          <th>Status</th>
          <th>GPU</th>
          <th>Parallelism</th>
          <th>Launch</th>
          <th>TPS</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${replicas.map((replica) => `
          <tr class="replica-row">
            <td>${escapeHtml(replica.replica_id)}</td>
            <td>
              <div class="replica-phase-cell">
                <span class="replica-phase-dot tone-${phaseTone(replica.phase)}"></span>
                <span>${escapeHtml(titleCase(replica.phase))}</span>
              </div>
            </td>
            <td>${escapeHtml(replica.gpu_type)}</td>
            <td>TP ${replica.tp} · PP ${replica.pp}</td>
            <td>${escapeHtml(titleCase(replica.launch_phase || replica.phase))}</td>
            <td>${formatNumber(replica.tps)}</td>
            <td>
              <div class="table-actions">
                <button type="button" class="secondary-button danger-button" data-action="kill" data-replica-id="${replica.replica_id}">Kill</button>
                <button type="button" class="secondary-button warning-button" data-action="throttle" data-replica-id="${replica.replica_id}">Throttle</button>
                <button type="button" class="secondary-button" data-action="restore" data-replica-id="${replica.replica_id}">Restore</button>
              </div>
            </td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderActivityConsole(snapshot) {
  if (!activityConsole) return;
  const runtimeEntries = ((snapshot.runtime && snapshot.runtime.events) || []).map((event) => {
    const description = describeRuntimeEvent(event);
    return {
      id: event.event_id || `${event.action}-${event.at_seconds}`,
      timestamp: (snapshot.created_at || 0) + (event.at_seconds || 0),
      source: "sim",
      kind: event.action || "event",
      title: description.title,
      detail: description.detail,
    };
  });

  const heartbeatPhases = new Set();
  const koiEntries = (((snapshot || {}).koi || {}).events || []).flatMap((event) => {
    const eventName = String(event.event || "");
    if (eventName === "job_launch_heartbeat") {
      const dedupeKey = `${event.job_id || ""}:${event.phase || ""}`;
      if (heartbeatPhases.has(dedupeKey)) {
        return [];
      }
      heartbeatPhases.add(dedupeKey);
    }

    const description = describeKoiEvent(event);
    return [{
      id: `${event.timestamp || 0}:${eventName}:${event.job_id || event.group_id || ""}:${event.phase || ""}`,
      timestamp: event.timestamp || 0,
      source: "koi",
      kind: eventName || "event",
      title: description.title,
      detail: description.detail,
    }];
  });

  const merged = [...runtimeEntries, ...koiEntries]
    .sort((a, b) => a.timestamp - b.timestamp)
    .slice(-28);

  if (!merged.length) {
    activityConsole.innerHTML = `<div class="console-empty">Launch a session to start the console.</div>`;
    return;
  }

  const shouldStick =
    (activityConsole.scrollTop + activityConsole.clientHeight) >= (activityConsole.scrollHeight - 48);

  activityConsole.innerHTML = merged.map((entry) => `
    <div class="console-line source-${entry.source} tone-${consoleToneForEntry(entry)}">
      <span class="console-time">${formatClock(entry.timestamp)}</span>
      <span class="console-source">${escapeHtml(entry.source)}</span>
      <div class="console-body">
        <span class="console-title">${escapeHtml(entry.title)}</span>
        ${entry.detail ? `<span class="console-text">${escapeHtml(entry.detail)}</span>` : ""}
      </div>
    </div>
  `).join("");

  if (shouldStick) {
    requestAnimationFrame(() => {
      activityConsole.scrollTop = activityConsole.scrollHeight;
    });
  }
}

async function loadCatalog() {
  const response = await fetch(previewMode ? "/demo/preview/catalog" : "/demo/catalog");
  renderCatalog(await response.json());
}

async function loadPreviewScene(sceneSlug) {
  const response = await fetch(`/demo/preview/scene/${encodeURIComponent(sceneSlug)}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Preview scene failed");
  }
  const snapshot = await response.json();
  renderSession(snapshot);
  launchStatus.textContent = "Preview mode";
  const next = new URL(window.location.href);
  next.searchParams.set("preview", "1");
  next.searchParams.set("scene", sceneSlug);
  window.history.replaceState({}, "", next);
}

async function postJson(url, payload = null) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: payload ? JSON.stringify(payload) : "{}",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

function startStream(sessionId) {
  if (activeSource) {
    activeSource.close();
  }

  let lockRefreshed = false;
  activeSource = new EventSource(`/demo/stream/${sessionId}`);
  activeSource.onmessage = (event) => {
    const snapshot = JSON.parse(event.data);
    renderSession(snapshot);
    launchStatus.textContent = snapshot.runtime.status === "koi_deciding"
      ? "Koi deciding..."
      : "Streaming";
    if (!lockRefreshed && ["completed", "launch_failed"].includes(snapshot.runtime.status)) {
      lockRefreshed = true;
      // Pick up the lock-released state once the live session ends.
      loadCatalog().catch(() => {});
    }
  };
  activeSource.onerror = () => {
    launchStatus.textContent = "Stream disconnected";
  };
}

async function launchSession(payload) {
  if (previewMode) {
    const scene = pageQuery.get("scene") || "running_healthy";
    await loadPreviewScene(scene);
    return;
  }
  launchStatus.textContent = "Creating session...";
  const response = await fetch("/demo/launch", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Launch failed");
  }
  const session = await response.json();
  renderSession(session);
  startStream(session.session_id);
  launchStatus.textContent = "Koi deciding...";
  // Refresh catalog so the Edit Quota lock state reflects the new live session.
  loadCatalog().catch(() => {});
}

function getReplicaById(replicaId) {
  const replicas = (((currentSnapshot || {}).runtime || {}).replicas) || [];
  return replicas.find((replica) => replica.replica_id === replicaId);
}

function getCurrentConfig(snapshot) {
  const decisionConfig = (((snapshot || {}).koi || {}).decision || {}).config || {};
  return {
    gpu_type: decisionConfig.gpu_type || snapshot.launch_preview.preferred_gpu,
    tp: decisionConfig.tp || snapshot.launch_preview.tp || 4,
    pp: decisionConfig.pp || snapshot.launch_preview.pp || 1,
  };
}

async function addReplica() {
  if (!currentSnapshot) return;
  if (previewMode) {
    previewAddReplica();
    launchStatus.textContent = "Preview replica added";
    return;
  }
  if (currentSnapshot.runtime.status === "koi_deciding") {
    launchStatus.textContent = "Wait for Koi to finish deciding first";
    return;
  }
  const cfg = getManualScaleConfig(currentSnapshot);
  launchStatus.textContent = "Adding replica...";
  const result = await postJson(`/demo/orca/job/${currentSnapshot.session_id}/scale`, {
    count: cfg.count,
    gpu_type: cfg.gpu_type,
    tp_size: cfg.tp,
    pp_size: cfg.pp,
    on_demand: true,
  });
  if (result.status !== "scaling") {
    throw new Error(result.message || result.reason || "Scale request failed");
  }
  launchStatus.textContent = `${cfg.count} ${cfg.gpu_type} replica${cfg.count > 1 ? "s" : ""} requested`;
}

async function killReplica(replicaId) {
  if (!currentSnapshot || !replicaId) return;
  if (previewMode) {
    previewKillReplica(replicaId);
    launchStatus.textContent = `${replicaId} removed`;
    return;
  }
  launchStatus.textContent = `Killing ${replicaId}...`;
  await postJson(`/demo/orca/job/${currentSnapshot.session_id}/kill`, {replica_ids: [replicaId]});
  launchStatus.textContent = `${replicaId} removed`;
}

async function setReplicaTps(replicaId, targetTps) {
  if (!currentSnapshot || !replicaId) return;
  if (previewMode) {
    previewSetReplicaTps(replicaId, targetTps);
    launchStatus.textContent = `${replicaId} now targets ${Math.round(targetTps)} tok/s`;
    return;
  }
  launchStatus.textContent = `Updating ${replicaId} TPS...`;
  await postJson(`/demo/orca/sim/set-tps/${replicaId}`, {target_tps: targetTps});
  launchStatus.textContent = `${replicaId} now targets ${Math.round(targetTps)} tok/s`;
}

function cloneSnapshot() {
  return JSON.parse(JSON.stringify(currentSnapshot));
}

function recalcPreviewSnapshot(snapshot) {
  const runtime = snapshot.runtime;
  const runningReplicas = runtime.replicas.filter((replica) => replica.phase === "running");
  runtime.active_replicas = runningReplicas.length;
  runtime.aggregate_tps = runningReplicas.reduce((sum, replica) => sum + Number(replica.tps || 0), 0);
  runtime.aggregate_tps = Number(runtime.aggregate_tps.toFixed(1));
  if (runtime.aggregate_tps <= 0) {
    runtime.status = "launching";
    runtime.eta_seconds = null;
    runtime.slo_headroom_pct = null;
  } else {
    runtime.status = "running";
    const remainingTokens = Math.max(0, Number(runtime.tokens_total || 0) - Number(runtime.tokens_completed || 0));
    runtime.eta_seconds = Number((remainingTokens / runtime.aggregate_tps).toFixed(1));
    const deadlineSeconds = Number((snapshot.request || {}).slo_deadline_hours || 0) * 3600;
    if (deadlineSeconds > 0) {
      const elapsed = Number(runtime.elapsed_seconds || 0);
      const projected = elapsed + Number(runtime.eta_seconds || 0);
      runtime.slo_headroom_pct = Number((((deadlineSeconds - projected) / deadlineSeconds) * 100).toFixed(1));
    }
  }
  return snapshot;
}

function pushPreviewConsoleEvent(snapshot, source, kind, title, detail = "") {
  const now = Date.now() / 1000;
  if (source === "koi") {
    snapshot.koi.events = snapshot.koi.events || [];
    const event = { event: kind, timestamp: now, job_id: snapshot.session_id, response: detail };
    if (kind === "tool_call") {
      event.label = "preview";
      event.call_number = 1;
      event.tool = title;
    } else if (kind === "trigger_handling") {
      event.trigger_type = title;
      delete event.response;
    } else if (kind === "job_started") {
      event.gpu_type = "L40S";
      event.tp = 4;
      event.pp = 2;
      event.response = detail;
    } else {
      event.response = detail || title;
    }
    snapshot.koi.events.push(event);
    snapshot.koi.events = snapshot.koi.events.slice(-20);
    return;
  }

  snapshot.runtime.events = snapshot.runtime.events || [];
  snapshot.runtime.events.push({
    event_id: `preview-${kind}-${Math.round(now * 1000)}`,
    at_seconds: Number(snapshot.runtime.elapsed_seconds || 0),
    action: kind,
    label: title,
    description: detail,
    params: {},
  });
  snapshot.runtime.events = snapshot.runtime.events.slice(-20);
}

function previewAddReplica() {
  if (!currentSnapshot) return;
  const snapshot = cloneSnapshot();
  const cfg = getManualScaleConfig(snapshot);
  const instance = getManualInstanceForGpu(snapshot, cfg.gpu_type);
  const quota = getManualQuotaForGpu(snapshot, cfg.gpu_type);
  const count = cfg.count;
  for (let i = 0; i < count; i += 1) {
    const replicaId = `${snapshot.session_id}-r${previewReplicaCounter}`;
    previewReplicaCounter += 1;
    snapshot.runtime.replicas.push({
      replica_id: replicaId,
      phase: "running",
      launch_phase: "running",
      gpu_type: cfg.gpu_type,
      instance_type: instance?.instance_type || snapshot.launch_preview.instance_type,
      tp: cfg.tp,
      pp: cfg.pp,
      region: quota?.region || snapshot.launch_preview.region,
      market: quota?.market || snapshot.launch_preview.market,
      tps: Number(snapshot.launch_preview.baseline_replica_tps || 0),
    });
  }
  pushPreviewConsoleEvent(snapshot, "sim", "scale_up", "Manual scale up", `Added ${count} ${cfg.gpu_type} replica${count > 1 ? "s" : ""} (TP ${cfg.tp}, PP ${cfg.pp}) in preview mode.`);
  currentSnapshot = recalcPreviewSnapshot(snapshot);
  renderSession(currentSnapshot);
}

function previewKillReplica(replicaId) {
  if (!currentSnapshot) return;
  const snapshot = cloneSnapshot();
  const replica = snapshot.runtime.replicas.find((item) => item.replica_id === replicaId);
  if (!replica) return;
  replica.phase = "killed";
  replica.launch_phase = "killed";
  replica.tps = 0;
  pushPreviewConsoleEvent(snapshot, "sim", "kill_replica", "Replica removed", `${replicaId} was removed in preview mode.`);
  pushPreviewConsoleEvent(snapshot, "koi", "trigger_handling", "failed", `Reacting to ${replicaId}.`);
  currentSnapshot = recalcPreviewSnapshot(snapshot);
  renderSession(currentSnapshot);
}

function previewSetReplicaTps(replicaId, targetTps) {
  if (!currentSnapshot) return;
  const snapshot = cloneSnapshot();
  const replica = snapshot.runtime.replicas.find((item) => item.replica_id === replicaId);
  if (!replica) return;
  replica.tps = Number(targetTps);
  pushPreviewConsoleEvent(snapshot, "sim", "set_replica_tps", "Replica TPS adjusted", `${replicaId} now targets ${Math.round(targetTps)} tok/s in preview mode.`);
  currentSnapshot = recalcPreviewSnapshot(snapshot);
  renderSession(currentSnapshot);
}

// ---------------------------------------------------------------------------
// Quota editor (persistent vCPU sliders per preset family/region/market)
// ---------------------------------------------------------------------------

function openQuotaEditor() {
  if (!catalog || !quotaEditor || !quotaEditorBody) return;
  const slug = quotaSelect.value;
  const preset = findQuotaPreset(slug);
  if (!preset) return;

  quotaEditorState.open = true;
  quotaEditorState.presetSlug = slug;

  // Working copy: start from stored overrides, fall back to defaults.
  const defaults = preset.defaults || {};
  const stored = preset.overrides || {};
  const working = {};
  (preset.editable_rows || []).forEach((row) => {
    const base = stored[row.key];
    working[row.key] = Number(base !== undefined ? base : (defaults[row.key] ?? row.default_vcpus));
  });
  quotaEditorState.workingValues = working;

  renderQuotaEditorBody();
  renderQuotaEditorLock(catalog.active_sessions);
  quotaEditorStatus.textContent = `Preset: ${preset.title || slug}`;
  quotaEditorStatus.classList.remove("is-error", "is-ok");
  quotaEditor.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeQuotaEditor() {
  quotaEditorState.open = false;
  if (quotaEditor) quotaEditor.hidden = true;
  document.body.style.overflow = "";
}

function renderQuotaEditorBody() {
  if (!quotaEditorBody) return;
  const preset = findQuotaPreset(quotaEditorState.presetSlug);
  if (!preset) {
    quotaEditorBody.innerHTML = "";
    return;
  }
  const rows = preset.editable_rows || [];
  if (!rows.length) {
    quotaEditorBody.innerHTML = '<div class="quota-empty">No adjustable quota rows for this preset.</div>';
    return;
  }
  quotaEditorBody.innerHTML = rows.map((row) => {
    const value = Number(quotaEditorState.workingValues[row.key] ?? row.default_vcpus);
    const marketLabel = String(row.market || "").replaceAll("_", " ");
    return `
      <div class="quota-editor-row" data-key="${escapeHtml(row.key)}">
        <div class="meta">
          <span class="title">${escapeHtml(row.family)}</span>
          <span class="caption">${escapeHtml(row.region)} · ${escapeHtml(marketLabel)} · default ${row.default_vcpus} vCPU</span>
        </div>
        <span class="value" data-quota-value>${value} vCPU</span>
        <input
          type="range"
          class="quota-editor-slider"
          data-quota-slider
          min="${row.min_vcpus}"
          max="${row.max_vcpus}"
          step="${row.step_vcpus}"
          value="${value}"
          ${quotaEditorState.locked ? "disabled" : ""}
        >
      </div>
    `;
  }).join("");

  quotaEditorBody.querySelectorAll("[data-quota-slider]").forEach((slider) => {
    slider.addEventListener("input", (event) => {
      const row = event.target.closest(".quota-editor-row");
      const key = row?.dataset?.key;
      if (!key) return;
      const value = Number(event.target.value);
      quotaEditorState.workingValues[key] = value;
      const display = row.querySelector("[data-quota-value]");
      if (display) display.textContent = `${value} vCPU`;
    });
  });
}

async function saveQuotaOverrides() {
  const slug = quotaEditorState.presetSlug;
  if (!slug) return;
  if (!quotaEditorSaveButton) return;
  quotaEditorSaveButton.disabled = true;
  quotaEditorStatus.textContent = "Saving…";
  quotaEditorStatus.classList.remove("is-error", "is-ok");
  try {
    const response = await fetch("/demo/quota/overrides", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({preset_slug: slug, overrides: quotaEditorState.workingValues}),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body?.detail || "Failed to save";
      throw new Error(detail);
    }
    quotaEditorState.locked = Boolean(body.locked);
    await loadCatalog();
    renderQuotaEditorLock(Number(body.active_sessions || 0));
    quotaEditorStatus.textContent = "Saved.";
    quotaEditorStatus.classList.add("is-ok");
    setTimeout(() => closeQuotaEditor(), 600);
  } catch (error) {
    quotaEditorStatus.textContent = error.message;
    quotaEditorStatus.classList.add("is-error");
  } finally {
    if (!quotaEditorState.locked) quotaEditorSaveButton.disabled = false;
  }
}

async function resetQuotaOverrides() {
  const slug = quotaEditorState.presetSlug;
  if (!slug) return;
  quotaEditorResetButton.disabled = true;
  quotaEditorStatus.textContent = "Resetting…";
  quotaEditorStatus.classList.remove("is-error", "is-ok");
  try {
    const response = await fetch(`/demo/quota/overrides/${encodeURIComponent(slug)}/reset`, {
      method: "POST",
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body?.detail || "Failed to reset");
    }
    await loadCatalog();
    const preset = findQuotaPreset(slug);
    (preset?.editable_rows || []).forEach((row) => {
      quotaEditorState.workingValues[row.key] = Number(row.default_vcpus);
    });
    renderQuotaEditorBody();
    renderQuotaEditorLock(Number(body.active_sessions || 0));
    quotaEditorStatus.textContent = "Reset to defaults.";
    quotaEditorStatus.classList.add("is-ok");
  } catch (error) {
    quotaEditorStatus.textContent = error.message;
    quotaEditorStatus.classList.add("is-error");
  } finally {
    if (!quotaEditorState.locked) quotaEditorResetButton.disabled = false;
  }
}

if (editQuotaButton) {
  editQuotaButton.addEventListener("click", openQuotaEditor);
}

if (quotaEditor) {
  quotaEditor.addEventListener("click", (event) => {
    if (event.target?.dataset?.quotaClose === "1") {
      closeQuotaEditor();
    }
  });
}

if (quotaEditorSaveButton) {
  quotaEditorSaveButton.addEventListener("click", saveQuotaOverrides);
}

if (quotaEditorResetButton) {
  quotaEditorResetButton.addEventListener("click", resetQuotaOverrides);
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && quotaEditorState.open) {
    closeQuotaEditor();
  }
});

quotaSelect.addEventListener("change", () => {
  renderSelectedQuota();
  syncManualScaleControls(currentSnapshot);
});
scenarioSelect.addEventListener("change", renderSelectedScenario);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    model_name: modelInput.value,
    quota_preset: quotaSelect.value,
    scenario: scenarioSelect.value,
    avg_input_tokens: Number(document.getElementById("input-tokens").value),
    avg_output_tokens: Number(document.getElementById("output-tokens").value),
    total_chunks: Number(document.getElementById("total-chunks").value),
    slo_deadline_hours: Number(document.getElementById("slo-hours").value),
    cost_cap_usd: Number(document.getElementById("cost-cap").value),
  };

  try {
    await launchSession(payload);
  } catch (error) {
    launchStatus.textContent = error.message;
  }
});

addReplicaButton.addEventListener("click", async () => {
  try {
    await addReplica();
  } catch (error) {
    launchStatus.textContent = error.message;
  }
});

killOldestButton.addEventListener("click", async () => {
  const running = (((currentSnapshot || {}).runtime || {}).replicas || []).find((replica) => replica.phase === "running");
  if (!running) {
    launchStatus.textContent = "No running replica to kill";
    return;
  }
  try {
    await killReplica(running.replica_id);
  } catch (error) {
    launchStatus.textContent = error.message;
  }
});

throttleButton.addEventListener("click", async () => {
  const running = (((currentSnapshot || {}).runtime || {}).replicas || []).find((replica) => replica.phase === "running");
  if (!running) {
    launchStatus.textContent = "No running replica to throttle";
    return;
  }
  try {
    await setReplicaTps(running.replica_id, 250);
  } catch (error) {
    launchStatus.textContent = error.message;
  }
});

if (replicaGrid) {
  replicaGrid.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const replicaId = button.dataset.replicaId;
    const action = button.dataset.action;
    const replica = getReplicaById(replicaId);
    if (!replica) return;

    try {
      if (action === "kill") {
        await killReplica(replicaId);
      } else if (action === "throttle") {
        await setReplicaTps(replicaId, 250);
      } else if (action === "restore") {
        await setReplicaTps(replicaId, currentSnapshot.launch_preview.baseline_replica_tps);
      }
    } catch (error) {
      launchStatus.textContent = error.message;
    }
  });
}

if (themeToggle) {
  bootstrapTheme();
  themeToggle.addEventListener("click", () => {
    const currentTheme = document.body.classList.contains("theme-dark") ? "dark" : "light";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    applyTheme(nextTheme);
  });
}

if (previewMode) {
  document.body.classList.add("preview-mode");
  form.querySelectorAll("input, select, button").forEach((element) => {
    element.disabled = true;
  });
  launchStatus.textContent = "Preview mode";

  loadCatalog()
    .then(async () => {
      const initialScene = pageQuery.get("scene") || "running_healthy";
      try {
        await loadPreviewScene(initialScene);
      } catch (error) {
        if (initialScene !== "running_healthy") {
          await loadPreviewScene("running_healthy");
          return;
        }
        throw error;
      }
    })
    .catch((error) => {
      launchStatus.textContent = error.message;
    });
} else {
  loadCatalog().catch((error) => {
    launchStatus.textContent = error.message;
  });
}
