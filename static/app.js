const state = { current: null, cursor: 0, total: 1, dataset: null };
// mode is the source of truth for which of the three ways a command is
// answered ("exact" | "search" | "policy"); `search` is kept in sync for the
// request body / UI toggle state the search-options panel reads.
// The LLM parser is the default (it handles phrasings the rule grammar can't,
// and falls back to the rule parser when Ollama is down), but the command is
// APPLIED exactly by default, not answered by the policy. Measured on
// "show only bones" (ct_chest): exact isolates it -- skeleton 16.7%, lungs
// 0.0% -- while the policy leaves soft tissue dominant at 43.1% and the
// skeleton at 4.9%. The policy's held-out attainment is +0.275 (median of
// three seeds; the seed-0 checkpoint this viewer loads is +0.231); it proposes
// a direction, it does not execute an instruction. Policy stays one click away.
const config = { parser: "llm", search: false, mode: "exact" };

const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>\"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));

function safeStorageGet(kind, key) {
  try {
    if (kind === "local") return localStorage.getItem(key);
    return sessionStorage.getItem(key);
  } catch (err) {
    return null;
  }
}

function safeStorageSet(kind, key, value) {
  try {
    if (kind === "local") localStorage.setItem(key, value);
    else sessionStorage.setItem(key, value);
  } catch (err) {
    // Storage can be unavailable in private or restricted browsing contexts.
  }
}

const THEME_KEY = "tf-rl-theme";

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = resolved;
  const button = el("theme-toggle");
  button?.setAttribute("aria-pressed", String(resolved === "dark"));
  button?.setAttribute("title", `Switch to ${resolved === "dark" ? "light" : "dark"} theme`);
  const label = button?.querySelector("#theme-label");
  if (label) label.textContent = `${resolved[0].toUpperCase()}${resolved.slice(1)} mode`;
}

function initTheme() {
  const saved = safeStorageGet("local", THEME_KEY);
  applyTheme(saved === "light" ? "light" : "dark");
  el("theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    safeStorageSet("local", THEME_KEY, next);
    applyTheme(next);
  });
}

function initAboutDialog() {
  const modal = el("about-modal");
  const closeButton = el("about-modal-close");
  if (!modal || !closeButton) return;

  let returnFocus = el("about-btn");
  const close = () => {
    if (modal.open) modal.close();
  };

  el("about-btn")?.addEventListener("click", () => {
    returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : el("about-btn");
    modal.showModal();
  });
  closeButton.addEventListener("click", close);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) close();
  });
  modal.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  });
  modal.addEventListener("cancel", (event) => {
    event.preventDefault();
    close();
  });
  modal.addEventListener("close", () => returnFocus?.focus());
}

initTheme();
initAboutDialog();

// The four goal classes, in a fixed order, shared by the chat replies and the
// comparison panel so a class keeps the same name and hue everywhere.
const CLASS_ORDER = ["skeleton", "lungs", "soft", "vessels"];
const CLASS_LABEL = { skeleton: "skeleton", lungs: "lungs", soft: "soft tissue", vessels: "vessels" };

const messagesEl = el("messages");
const emptyState = el("empty-state");
const textInput = el("text-input");
const sendBtn = el("send-btn");
const loadingOverlay = el("loading-overlay");
let loadingCount = 0;

function setLoading(loading) {
  loadingCount = Math.max(0, loadingCount + (loading ? 1 : -1));
  const active = loadingCount > 0;
  loadingOverlay.hidden = !active;
  loadingOverlay.setAttribute("aria-busy", String(active));
}

async function withLoading(task) {
  setLoading(true);
  try {
    return await task();
  } finally {
    setLoading(false);
  }
}

let sceneSnapshot = null;
let lastState = null;
const sceneNonce = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const sceneSequenceKey = "localViewerSceneSequence";
const storedSceneSequence = Number.parseInt(safeStorageGet("session", sceneSequenceKey) || "0", 10);
let sceneSequence = Number.isSafeInteger(storedSceneSequence) && storedSceneSequence >= 0
  ? storedSceneSequence
  : 0;

function nextSceneId(data, suffix = "") {
  sceneSequence += 1;
  safeStorageSet("session", sceneSequenceKey, String(sceneSequence));
  return `web:${data.session_id || "web-session"}:${sceneNonce}:scene:${sceneSequence}${suffix}`;
}

function transitionIdentity(data, before, metadata = {}) {
  return encodeURIComponent(JSON.stringify({
    nonce: sceneNonce,
    session_id: data.session_id,
    parent_scene_id: before.scene_id,
    step_id: data.current.id,
    command: data.current.cmd_dict,
    params: data.current.params,
    camera: data.current.camera,
    metadata,
  }));
}

function cameraForScene(current) {
  const camera = window.volumeViewer?.getCamera();
  if (camera?.position && camera?.focal_point && camera?.view_up) return camera;
  return {
    position: [0, 0, 1], focal_point: [0, 0, 0], view_up: [0, 1, 0],
    zoom: current.camera?.zoom || 1,
  };
}

function sceneFromState(data, parentSceneId = null, sceneId = null) {
  const current = data.current;
  const rawCommand = current.cmd_dict || {};
  const isOpacity = rawCommand.attribute === "opacity" &&
    typeof rawCommand.target === "string" &&
    (rawCommand.direction === "increase" || rawCommand.direction === "decrease");
  const command = rawCommand.camera
    ? { attribute: "camera" }
    : isOpacity
      ? { attribute: "opacity", target: rawCommand.target, direction: rawCommand.direction }
      : { attribute: "neutral", kind: "non_extractable" };
  const scene = {
    scene_id: sceneId || nextSceneId(data),
    parent_scene_id: parentSceneId,
    step_id: data.current.id,
    parent_step_id: parentSceneId ? (sceneSnapshot?.step_id ?? null) : null,
    carried_forward: Boolean(parentSceneId),
    session_id: data.session_id || "web-session",
    client: "web",
    dataset: data.dataset,
    dataset_version: data.render_info?.dataset_version || `${data.dataset}-unknown`,
    volume: {
      dimensions: data.render_info?.volume_shape || [1, 1, 1],
      spacing: data.render_info?.spacing || [1, 1, 1],
      scalar_type: "float32",
      orientation: "dataset-normalized",
    },
    transfer_function: current.params,
    camera: cameraForScene(current),
    command,
    client_metadata: {
      source: "static-app", cursor: data.cursor, total: data.total,
      ...(isOpacity || rawCommand.camera ? {} : { original_command: rawCommand }),
    },
    features_before: lastState?.current?.features || null,
    features_after: current.features || null,
  };
  if (command.attribute === "opacity") {
    scene.goal = { target: command.target, direction: command.direction };
  }
  return scene;
}

function captureSceneRoot(data) {
  sceneSnapshot = sceneFromState(data, null);
  sceneSnapshot.command = { attribute: "neutral", kind: "root" };
  delete sceneSnapshot.goal;
  sceneSnapshot.scene_id = `web:${data.session_id || "web-session"}:${sceneNonce}:root`;
  lastState = data;
}

async function postSceneTransition(data, metadata = {}) {
  if (!sceneSnapshot) captureSceneRoot(data);
  const before = sceneSnapshot;
  const identity = transitionIdentity(data, before, metadata);
  const after = { ...sceneFromState(data, sceneSnapshot.scene_id, `web:${data.session_id}:${sceneNonce}:transition:${identity}`), ...metadata };
  const eventId = `web:${data.session_id}:${sceneNonce}:event:${identity}`;
  const response = await fetch("/api/scenes/transition", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ before, after, event_id: eventId, dedupe_key: eventId }),
  });
  if (!response.ok) {
    const detail = await response.text();
    showToast(`Scene transition failed (${response.status}): ${detail || "server rejected record"}`, "destructive");
    return;
  }
  sceneSnapshot = await response.json();
  lastState = data;
}

async function refresh(data) {
  state.current = data.current;
  state.cursor = data.cursor;
  state.total = data.total;
  if (data.dataset && data.dataset !== state.dataset) {
    setSelectValue(data.dataset);
  }

  el("step-counter").textContent = `${state.cursor + 1} / ${state.total}`;
  el("back-btn").disabled = state.cursor <= 0;
  el("forward-btn").disabled = state.cursor >= state.total - 1;

  el("current-image").src = `data:image/png;base64,${state.current.image_b64}`;
  if (window.volumeViewer && state.dataset && state.current) {
    await window.volumeViewer.load(state.dataset, state.current.params, state.current.camera);
  }

  updateTelemetry(state.current.class_visibility);
  return data;
}

// The readout is per-goal-class visibility (share of the rendered image), the
// same quantity the policy is scored on -- not opacity mass over the retired
// fat/air/spongy bands. A class this volume cannot support (vessels without
// contrast) arrives as null and stays a dash.
function updateTelemetry(classVisibility) {
  if (!classVisibility) return;
  for (const goalClass of ["skeleton", "lungs", "soft", "vessels"]) {
    const v = classVisibility[goalClass];
    const node = el(`telem-${goalClass}`);
    if (!node) continue;
    node.textContent = (v === undefined || v === null) ? "—" : `${(v * 100).toFixed(1)}%`;
  }
}

function rebuildMessages(history) {
  messagesEl.innerHTML = "";
  const withText = history.filter((s) => s.cmd_text);
  if (withText.length === 0) {
    messagesEl.appendChild(emptyState);
    return;
  }
  withText.forEach(appendMessage);
}

function appendMessage(step) {
  if (emptyState.parentNode === messagesEl) messagesEl.removeChild(emptyState);

  const div = document.createElement("div");
  div.className = "msg";

  const text = document.createElement("div");
  text.className = "msg-text";
  text.textContent = step.cmd_text;
  div.appendChild(text);

  const tagNames = [];
  if (step.mode === "policy") tagNames.push("policy");
  else if (step.search) tagNames.push("search");
  // parser_used, not the toggle: an llm request that fell back must not be
  // labelled "llm".
  if (step.parser_used) tagNames.push(step.parser_used);
  if (tagNames.length) {
    const tags = document.createElement("div");
    tags.className = "msg-tags";
    for (const name of tagNames) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = name;
      tags.appendChild(t);
    }
    div.appendChild(tags);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// The system's turn in the conversation. It does not chat -- it reports what
// the render now shows, which is the measurement the whole project is built
// on. Without it the thread is a list of things the user said, and the effect
// of each command is invisible.
const METHOD_SAID = {
  policy: "the policy answered",
  search: "search answered",
  exact: "applied directly",
  camera: "moved the camera",
};

function appendReply(step, before) {
  const div = document.createElement("div");
  div.className = "msg-reply";

  const method = document.createElement("div");
  method.className = "reply-method";
  method.textContent = METHOD_SAID[step.mode] || METHOD_SAID.exact;
  div.appendChild(method);

  // A fallback or a search that found nothing is the most useful thing the
  // system can say, so it belongs in the thread, not only in a toast that
  // disappears.
  if (step.message) {
    const note = document.createElement("div");
    note.className = "reply-note";
    note.textContent = step.message;
    div.appendChild(note);
  }

  const now = step.class_visibility || {};
  const moved = CLASS_ORDER.filter((c) => {
    const a = before ? before[c] : null;
    const b = now[c];
    return a != null && b != null && Math.abs(b - a) >= 0.0001;
  });

  if (moved.length) {
    const list = document.createElement("div");
    list.className = "reply-deltas";
    for (const c of moved) {
      const row = document.createElement("div");
      row.className = "reply-delta";
      row.innerHTML = `<span class="reply-dot" style="background:var(--class-${c})"></span>
        <span class="reply-class">${CLASS_LABEL[c]}</span>
        <span class="reply-numbers">${(before[c] * 100).toFixed(2)} &rarr; ${(now[c] * 100).toFixed(2)}%</span>`;
      div.appendChild(list);
      list.appendChild(row);
    }
  } else if (!step.message) {
    const none = document.createElement("div");
    none.className = "reply-note";
    none.textContent = "Nothing moved measurably.";
    div.appendChild(none);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function loadState() {
  const r = await fetch("/api/state");
  const data = await r.json();
  await refresh(data);
  captureSceneRoot(data);
  sceneSnapshot.command = { attribute: "neutral", kind: "dataset_boundary" };
  // Full history isn't in /api/state (only the current step), so rebuild the
  // thread by walking back/forward would be wasteful; instead the server's
  // current step is enough to seed the empty state on first load.
  if (data.current.cmd_text) {
    messagesEl.innerHTML = "";
    appendMessage(data.current);
  }
}

function sendCommand(text) {
  return withLoading(() => sendCommandImpl(text));
}

async function sendCommandImpl(text) {
  // What the render showed before this command, so the reply can report the
  // change rather than only the new value.
  const before = state.current ? state.current.class_visibility : null;
  const body = {
    text,
    parser: config.parser,
    search: config.search,
    steps: parseInt(el("steps-input").value, 10),
    mode: config.mode,
  };
  const r = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.status !== 200) {
    const err = await r.json();
    showToast(`Could not parse that: ${err.detail}`, "destructive");
    return;
  }
  const data = await r.json();
  await refresh(data);
  await postSceneTransition(data);
  appendMessage(data.current);
  appendReply(data.current, before);
}

function autoResize() {
  textInput.style.height = "auto";
  textInput.style.height = `${Math.min(textInput.scrollHeight, 136)}px`;
}

function updateSendState() {
  sendBtn.disabled = textInput.value.trim().length === 0;
}

function submitText() {
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  autoResize();
  updateSendState();
  sendCommand(text);
}

sendBtn.addEventListener("click", submitText);
textInput.addEventListener("input", () => { autoResize(); updateSendState(); });
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitText();
  }
});

function navigate(path) {
  return withLoading(() => navigateImpl(path));
}

async function navigateImpl(path) {
  const previous = lastState;
  const data = await (await fetch(path, { method: "POST" })).json();
  await refresh(data);
  await postSceneTransition(data, {
    carried_forward: true,
    parent_step_id: previous?.current?.id ?? null,
  });
}

el("back-btn").addEventListener("click", () => navigate("/api/back"));
el("forward-btn").addEventListener("click", () => navigate("/api/forward"));
el("reset-btn").addEventListener("click", () => sendCommand("reset"));

// ---------- toolbar: toggle groups + single toggle ----------

// A ToggleGroup is a single-select set of buttons sharing one `data-toggle-group`
// container: exactly one child carries data-state="on" at a time, clicking a
// sibling flips it, and the container's mapped `config` key is kept in sync.
function initToggleGroup(container) {
  const key = container.dataset.toggleGroup;
  const items = Array.from(container.querySelectorAll(".toggle-item"));
  const selectItem = (selected) => {
    items.forEach((other) => {
      const active = other === selected;
      other.dataset.state = active ? "on" : "off";
      other.setAttribute("aria-checked", String(active));
      other.tabIndex = active ? 0 : -1;
    });
    config[key] = selected.dataset.value;
  };
  items.forEach((item) => {
    item.setAttribute("role", "radio");
    item.setAttribute("aria-checked", item.dataset.state === "on" ? "true" : "false");
    item.tabIndex = item.dataset.state === "on" ? 0 : -1;
    item.addEventListener("click", () => selectItem(item));
    item.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", " ", "Enter"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === " " || event.key === "Enter") {
        selectItem(item);
        return;
      }
      const current = items.indexOf(item);
      const next = event.key === "Home" ? 0
        : event.key === "End" ? items.length - 1
          : (current + ((event.key === "ArrowLeft" || event.key === "ArrowUp") ? -1 : 1) + items.length) % items.length;
      items[next].focus();
      selectItem(items[next]);
    });
  });
}

document.querySelectorAll("[data-toggle-group]").forEach(initToggleGroup);

const searchToggleBtn = el("search-toggle-btn");
const searchOptions = el("search-options");
const policyToggleBtn = el("policy-toggle-btn");

// search and policy are two different ways of answering the same command --
// mutually exclusive, alongside the default "exact" mode.
searchToggleBtn.addEventListener("click", () => {
  config.search = !config.search;
  searchToggleBtn.dataset.active = String(config.search);
  searchToggleBtn.setAttribute("aria-pressed", String(config.search));
  searchOptions.hidden = !config.search;
  if (config.search) {
    policyToggleBtn.dataset.active = "false";
    policyToggleBtn.setAttribute("aria-pressed", "false");
    config.mode = "search";
  } else {
    config.mode = "exact";
  }
});

policyToggleBtn.addEventListener("click", () => {
  const active = policyToggleBtn.dataset.active !== "true";
  policyToggleBtn.dataset.active = String(active);
  policyToggleBtn.setAttribute("aria-pressed", String(active));
  if (active) {
    config.search = false;
    searchToggleBtn.dataset.active = "false";
    searchToggleBtn.setAttribute("aria-pressed", "false");
    searchOptions.hidden = true;
    config.mode = "policy";
  } else {
    config.mode = "exact";
  }
});

// ---------- mic: push to talk + real audio-reactive waveform ----------

const micBtn = el("mic-btn");
const micWaveform = el("mic-waveform");
const micIcon = el("mic-icon");
const waveformBars = Array.from(micWaveform.querySelectorAll("span"));

let mediaRecorder = null;
let chunks = [];
let audioCtx = null;
let analyser = null;
let waveformRAF = null;

function startWaveform(stream) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  source.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  const tick = () => {
    analyser.getByteFrequencyData(data);
    const step = Math.floor(data.length / waveformBars.length);
    waveformBars.forEach((bar, i) => {
      const v = data[i * step] / 255; // 0..1
      bar.style.height = `${4 + v * 14}px`;
    });
    waveformRAF = requestAnimationFrame(tick);
  };
  tick();
}

function stopWaveform() {
  if (waveformRAF) cancelAnimationFrame(waveformRAF);
  waveformRAF = null;
  if (audioCtx) audioCtx.close();
  audioCtx = null;
  waveformBars.forEach((bar) => { bar.style.height = "4px"; });
}

async function startRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showToast("Microphone access denied or unavailable", "destructive");
    return;
  }
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = async () => {
    micBtn.classList.remove("recording");
    micWaveform.hidden = true;
    micIcon.style.display = "";
    stopWaveform();
    stream.getTracks().forEach((t) => t.stop());

    const blob = new Blob(chunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "clip.webm");
    micBtn.disabled = true;
    try {
      await withLoading(async () => {
        const r = await fetch("/api/transcribe", { method: "POST", body: form });
        const data = await r.json();
        if (data.text) sendCommand(data.text);
      });
    } finally {
      micBtn.disabled = false;
    }
  };
  mediaRecorder.start();
  micBtn.classList.add("recording");
  micIcon.style.display = "none";
  micWaveform.hidden = false;
  startWaveform(stream);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
}

micBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") stopRecording();
  else startRecording();
});

// Push-to-talk hotkey: hold Space to record, release to stop. Skipped while
// the text input is focused so typing a literal space still works there.
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || document.activeElement === textInput) return;
  e.preventDefault();
  startRecording();
});

document.addEventListener("keyup", (e) => {
  if (e.code !== "Space" || document.activeElement === textInput) return;
  e.preventDefault();
  stopRecording();
});

// ---------- dataset selector (custom Select: trigger button + listbox popover) ----------

const selectTrigger = el("dataset-select-trigger");
const selectValueLabel = el("dataset-select-value");
const selectPopover = el("dataset-select-popover");
let selectOptions = [];
let selectHighlighted = -1;

function setSelectValue(name) {
  state.dataset = name;
  selectValueLabel.textContent = name;
  selectOptions.forEach((opt) => opt.setAttribute("aria-selected", String(opt.dataset.value === name)));
}

function openSelect() {
  selectPopover.hidden = false;
  selectTrigger.setAttribute("aria-expanded", "true");
  selectHighlighted = selectOptions.findIndex((opt) => opt.dataset.value === state.dataset);
  highlightSelectOption(selectHighlighted);
}

function closeSelect() {
  selectPopover.hidden = true;
  selectTrigger.setAttribute("aria-expanded", "false");
}

function highlightSelectOption(index) {
  selectOptions.forEach((opt, i) => opt.classList.toggle("highlighted", i === index));
  if (index >= 0) selectOptions[index].scrollIntoView({ block: "nearest" });
}

function chooseDataset(name) {
  return withLoading(() => chooseDatasetImpl(name));
}

async function chooseDatasetImpl(name) {
  if (name === state.dataset) { closeSelect(); return; }
  const previous = state.dataset;
  const previousSceneId = sceneSnapshot?.scene_id || "none";
  const r = await fetch("/api/dataset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (r.status !== 200) {
    const err = await r.json();
    showToast(`Could not switch dataset: ${err.detail}`, "destructive");
    setSelectValue(previous);
    closeSelect();
    return;
  }
  const data = await r.json();
  setSelectValue(name);
  sceneSnapshot = null;
  lastState = null;
  closeSelect();
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  await refresh(data);
  captureSceneRoot(data);
  const boundaryEventId = `web:${data.session_id}:${sceneNonce}:boundary:${previousSceneId}:${sceneSnapshot.scene_id}:${data.dataset}:${data.render_info.dataset_version}`;
  const boundaryResponse = await fetch("/api/scenes/transition", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      after: { ...sceneSnapshot, command: { attribute: "neutral", kind: "dataset_boundary" } },
      boundary: true,
       event_id: boundaryEventId,
       dedupe_key: boundaryEventId,
    }),
  });
  if (boundaryResponse.ok) sceneSnapshot = await boundaryResponse.json();
}

async function loadDatasets() {
  const r = await fetch("/api/datasets");
  const data = await r.json();
  selectPopover.innerHTML = "";
  selectOptions = data.available.map((name) => {
    const opt = document.createElement("div");
    opt.className = "select-option";
    opt.setAttribute("role", "option");
    opt.dataset.value = name;
    opt.textContent = name;
    opt.addEventListener("click", () => chooseDataset(name));
    selectPopover.appendChild(opt);
    return opt;
  });
  setSelectValue(data.current);
}

selectTrigger.addEventListener("click", () => {
  if (selectPopover.hidden) openSelect(); else closeSelect();
});

const SELECT_NAV_KEYS = ["ArrowDown", "ArrowUp", "Enter", " "];

selectTrigger.addEventListener("keydown", (e) => {
  if (SELECT_NAV_KEYS.includes(e.key)) e.preventDefault();
  if (selectPopover.hidden) {
    if (SELECT_NAV_KEYS.includes(e.key)) openSelect();
    return;
  }
  if (e.key === "ArrowDown") {
    selectHighlighted = Math.min(selectHighlighted + 1, selectOptions.length - 1);
    highlightSelectOption(selectHighlighted);
  } else if (e.key === "ArrowUp") {
    selectHighlighted = Math.max(selectHighlighted - 1, 0);
    highlightSelectOption(selectHighlighted);
  } else if ((e.key === "Enter" || e.key === " ") && selectHighlighted >= 0) {
    chooseDataset(selectOptions[selectHighlighted].dataset.value);
  } else if (e.key === "Escape") {
    closeSelect();
  }
});

document.addEventListener("click", (e) => {
  if (!selectPopover.hidden && !el("dataset-select").contains(e.target)) closeSelect();
});

// ---------- command reference modal ----------

let commandsCache = null;
let commandsReturnFocus = el("commands-help-btn");
let commandsOpenPromise = null;

async function openCommandsModal() {
  const modal = el("commands-modal");
  if (modal.open || commandsOpenPromise) return;
  commandsReturnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : el("commands-help-btn");
  commandsOpenPromise = (async () => {
    if (!commandsCache) {
      const res = await fetch("/api/commands");
      const data = await res.json();
      commandsCache = data.commands;
      const list = el("commands-list");
      list.innerHTML = "";
      for (const entry of commandsCache) {
        const section = document.createElement("div");
        section.className = "cmd-category";
        const title = document.createElement("h3");
        title.textContent = entry.category;
        const desc = document.createElement("p");
        desc.textContent = entry.description;
        const examples = document.createElement("div");
        examples.className = "cmd-examples";
        for (const example of entry.examples) {
          const chip = document.createElement("code");
          chip.className = "cmd-example-chip";
          chip.textContent = example;
          examples.appendChild(chip);
        }
        section.appendChild(title);
        section.appendChild(desc);
        section.appendChild(examples);
        list.appendChild(section);
      }
    }
    modal.showModal();
    // Dialog starts at opacity/scale 0 in CSS; adding this class one frame
    // later is what makes the transition to full opacity/scale actually run
    // (toggling it in the same frame as showModal() would skip straight to
    // the end state with no visible animation).
    requestAnimationFrame(() => modal.classList.add("dialog-open"));
  })();
  try {
    await commandsOpenPromise;
  } finally {
    commandsOpenPromise = null;
  }
}

function closeCommandsModal() {
  const modal = el("commands-modal");
  if (modal.open) modal.close();
}

const modal = el("commands-modal");
el("commands-help-btn").addEventListener("click", openCommandsModal);
el("commands-modal-close").addEventListener("click", closeCommandsModal);
modal.addEventListener("click", (e) => {
  if (e.target === modal) closeCommandsModal();
});
modal.addEventListener("cancel", () => {
  // Native Escape closes the dialog; close handles shared cleanup.
});
modal.addEventListener("close", () => {
  modal.classList.remove("dialog-open");
  commandsReturnFocus?.focus();
});

// ---------- tooltip ----------
// One shared floating element, positioned above whichever [data-tooltip]
// element is currently hovered/focused -- mirrors shadcn's Tooltip (a
// single Radix popper instance reused per trigger, not one DOM node per
// trigger).

const tooltipEl = document.createElement("div");
tooltipEl.id = "tooltip";
tooltipEl.hidden = true;
document.body.appendChild(tooltipEl);

let tooltipShowTimer = null;

function positionTooltip(trigger) {
  const rect = trigger.getBoundingClientRect();
  const tipRect = tooltipEl.getBoundingClientRect();
  const left = rect.left + rect.width / 2 - tipRect.width / 2;
  const top = rect.top - tipRect.height - 8;
  tooltipEl.style.left = `${Math.max(4, Math.min(left, window.innerWidth - tipRect.width - 4))}px`;
  tooltipEl.style.top = `${Math.max(4, top)}px`;
}

function showTooltip(trigger) {
  const text = trigger.dataset.tooltip;
  if (!text) return;
  tooltipShowTimer = setTimeout(() => {
    tooltipEl.textContent = text;
    tooltipEl.hidden = false;
    positionTooltip(trigger);
  }, 400);
}

function hideTooltip() {
  clearTimeout(tooltipShowTimer);
  tooltipEl.hidden = true;
}

document.querySelectorAll("[data-tooltip]").forEach((trigger) => {
  trigger.addEventListener("mouseenter", () => showTooltip(trigger));
  trigger.addEventListener("mouseleave", hideTooltip);
  trigger.addEventListener("focus", () => showTooltip(trigger));
  trigger.addEventListener("blur", hideTooltip);
  trigger.addEventListener("click", hideTooltip);
});

// ---------- toast ----------

const toastViewport = el("toast-viewport");
const TOAST_DURATION_MS = 5000;

function showToast(message, variant = "default") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${variant}`;
  toast.setAttribute("role", variant === "destructive" ? "alert" : "status");

  const text = document.createElement("span");
  text.className = "toast-text";
  text.textContent = message;
  toast.appendChild(text);

  const closeBtn = document.createElement("button");
  closeBtn.className = "toast-close";
  closeBtn.setAttribute("aria-label", "Dismiss");
  closeBtn.textContent = "×";
  toast.appendChild(closeBtn);

  const dismiss = () => {
    toast.classList.add("toast-leaving");
    toast.addEventListener("transitionend", () => toast.remove(), { once: true });
  };
  closeBtn.addEventListener("click", dismiss);
  setTimeout(dismiss, TOAST_DURATION_MS);

  toastViewport.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("toast-visible"));
}

updateSendState();
loadDatasets();
loadState();

// --- four-mode comparison -----------------------------------------------------
// Answers the instruction four ways from the current step without advancing the
// session, so the amortisation claim -- the policy reaching search's
// neighbourhood while spending no visibility evaluations -- is on one screen
// instead of spread across four interactions the viewer has to hold in memory.

const ARM_ORDER = ["exact", "search_cheap", "search_thorough", "policy"];
const ARM_LABEL = {
  exact: "exact",
  search_cheap: "search · 10",
  search_thorough: "search · 200",
  policy: "policy",
};

const compareView = el("compare-view");
const compareColumns = el("compare-columns");
const compareGoal = el("compare-goal");

async function showCompare(on) {
  el("single-view").hidden = on;
  compareView.hidden = !on;
  if (on) return;

  // The viewport is a WebGL canvas. Hiding it with `hidden` is display:none,
  // so it measures 0x0 while the panel is open and comes back blank: vtk
  // sized itself to nothing and has no reason to redraw. Nudge the layout,
  // then re-issue the current step so both the canvas and the fallback image
  // are painted from real state rather than whatever survived being hidden.
  window.dispatchEvent(new Event("resize"));
  try {
    // Fresh state, not `lastState`: back/forward move the cursor without
    // updating it, so restoring from the cached copy could quietly rewind the
    // view to a different step than the one the counter shows.
    const r = await fetch("/api/state");
    if (r.ok) await refresh(await r.json());
  } catch (err) {
    showToast("Could not redraw the view. Step back and forward to restore it.", "destructive");
  }
  if (window.volumeViewer && window.volumeViewer.render) window.volumeViewer.render();
}

const pct = (v) => `${Math.min(100, Math.max(0, (v || 0) * 100)).toFixed(1)}%`;

function armSkeleton(name) {
  return `<div class="card compare-arm">
    <span class="compare-arm-name">${escapeHtml(ARM_LABEL[name])}</span>
    <div class="skeleton compare-arm-image"></div>
    <div class="skeleton" style="height:1.45rem;width:5rem"></div>
    <div class="skeleton" style="height:0.72rem;width:7rem"></div>
  </div>`;
}

// `goals.distance` scores two channels -- how much of the image a class
// contributes (vis) and how bright it appears (bright) -- and an instruction
// names one or the other. Plotting visibility for "brighten the skeleton"
// shows four columns with near-identical bars and attainment from 0.00 to
// 0.94, which reads as the panel contradicting itself. So plot the channel
// the instruction is actually about, and say which one.
const CHANNEL = {
  vis: { label: "visible", scale: (v) => v * 100, unit: "%", format: (v) => `${(v * 100).toFixed(1)}%` },
  bright: { label: "brightness", scale: (v) => v * 100, unit: "", format: (v) => v.toFixed(3) },
};

function classBars(values, start, channel) {
  const spec = CHANNEL[channel];
  return `<div class="compare-channel">${escapeHtml(spec.label)} per class</div>
  <table class="table">${CLASS_ORDER.map((c) => {
    const value = values && values[c] != null ? values[c] : null;
    const from = start && start[c] != null ? start[c] : 0;
    if (value === null) {
       return `<tr><td class="table-label">${escapeHtml(CLASS_LABEL[c])}</td>
        <td colspan="2" class="table-value">&mdash;</td></tr>`;
    }
    return `<tr>
       <td class="table-label">${escapeHtml(CLASS_LABEL[c])}</td>
      <td style="width:100%">
        <div class="progress">
          <div class="progress-fill" style="width:${pct(spec.scale(value) / 100)};--progress-color:var(--class-${c})"></div>
          <div class="progress-tick" style="left:${pct(spec.scale(from) / 100)}"></div>
        </div>
      </td>
      <td class="table-value">${escapeHtml(spec.format(value))}</td>
    </tr>`;
  }).join("")}</table>`;
}

function armCard(name, arm, start, channel) {
  if (!arm || arm.unavailable) {
    return `<div class="card compare-arm">
      <span class="compare-arm-name">${escapeHtml(ARM_LABEL[name])}</span>
      <p class="compare-unavailable">${escapeHtml(arm ? arm.unavailable : "no result")}</p>
    </div>`;
  }
  const sign = arm.attainment >= 0 ? "positive" : "negative";
  const shown = `${arm.attainment >= 0 ? "+" : "−"}${Math.abs(arm.attainment).toFixed(3)}`;
  const evals = `${arm.evaluations} eval${arm.evaluations === 1 ? "" : "s"}`;
  // An arm that returned its own input did not move. Without saying so, the
  // column reads as "this arm agrees with the start" -- the opposite claim.
  const noMove = arm.unchanged
    ? `<div class="compare-nomove">did not move &mdash; ${escapeHtml(evals)} found no improving step</div>`
    : "";
  return `<div class="card compare-arm">
    <span class="compare-arm-name">${escapeHtml(ARM_LABEL[name])}</span>
    <img class="compare-arm-image" src="data:image/png;base64,${escapeHtml(arm.image_b64)}" alt="${escapeHtml(ARM_LABEL[name])} result" />
    <div>
      <div class="compare-attainment" data-sign="${escapeHtml(sign)}">${escapeHtml(shown)}</div>
      <div class="compare-attainment-label">ATTAINMENT</div>
    </div>
    <div class="compare-cost">${escapeHtml(evals)} · ${escapeHtml(arm.elapsed_ms)} ms</div>
    ${noMove}
    <hr class="separator" />
    ${classBars(channel === "bright" ? arm.class_brightness : arm.class_visibility, start, channel)}
  </div>`;
}

function runCompare() {
  return withLoading(() => runCompareImpl());
}

async function runCompareImpl() {
  // `submitText` clears the composer, so after sending an instruction the box
  // is empty -- and "type it, send it, then compare it" is the natural flow.
  // Falling back to the step's own instruction is what makes the button work
  // when a reader expects it to.
  const typed = textInput.value.trim();
  const applied = lastState && lastState.current ? lastState.current.cmd_text : null;
  const text = typed || applied;
  if (!text) {
    showToast("Type an instruction, then press compare to answer it four ways.");
    return;
  }
  if (!typed) {
    // The comparison starts from where the session is now. If that instruction
    // has already been applied, this compares from *after* it, which is not
    // the same question -- say so rather than quietly answering a different one.
    showToast(`Comparing "${text}" from the current step. Step back first to compare it from where it was answered.`);
  }
  showCompare(true);
  compareGoal.innerHTML = `<strong>"${escapeHtml(text)}"</strong>`;
  compareColumns.innerHTML = ARM_ORDER.map(armSkeleton).join("");

  let payload;
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, parser: config.parser }),
    });
    payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "compare failed");
  } catch (err) {
    compareColumns.innerHTML = `<p class="compare-unavailable">${escapeHtml(err.message)}</p>`;
    return;
  }

  if (!payload.applicable) {
    compareColumns.innerHTML =
      `<p class="compare-unavailable">${escapeHtml(payload.reason)}<br />The comparison answers visibility instructions; camera and reset commands have nothing to score.</p>`;
    return;
  }

  compareGoal.innerHTML = `<strong>"${escapeHtml(payload.text)}"</strong> &rarr; ${escapeHtml(payload.goal_text)}`;
  // If the instruction names brightness at all, that is the channel it is
  // about -- otherwise visibility.
  const channel = payload.channels && payload.channels.bright.length ? "bright" : "vis";
  const start = payload.start
    ? (channel === "bright" ? payload.start.class_brightness : payload.start.class_visibility)
    : null;
  compareColumns.innerHTML = ARM_ORDER.map((name) => armCard(name, payload.arms[name], start, channel)).join("");
}

el("compare-btn").addEventListener("click", runCompare);
el("compare-close").addEventListener("click", () => { showCompare(false); });

// --- the sweep: the distribution, not one draw --------------------------------
// One comparison is a single episode. The claim the thesis makes is a median
// over many, and the reliability gap means about one instruction in four has
// the policy not improving -- so a single press invites generalising from an
// anecdote, in either direction. This runs many sampled instructions and shows
// the spread.

const SWEEP_ARMS = ["exact", "search_cheap", "policy"];

function sweepRow(name, arm, best) {
  if (!arm || arm.n === 0 || !Number.isFinite(arm.median)) {
    if (name !== "policy") {
      return `<tr><td class="table-label">${escapeHtml(ARM_LABEL[name])}</td>
        <td colspan="3" class="table-value">&mdash;</td></tr>`;
    }
    const reason = arm?.unavailable || "policy arm has no checkpoint-backed samples";
    return `<tr><td class="table-label">${escapeHtml(ARM_LABEL[name])}</td>
      <td colspan="3" class="table-value"><strong>Policy unavailable</strong><br />${escapeHtml(reason)}</td></tr>`;
  }
  const median = arm.median;
  const width = Math.min(100, Math.max(0, (median / best) * 100));
  const sign = median >= 0 ? "positive" : "negative";
  return `<tr>
    <td class="table-label">${escapeHtml(ARM_LABEL[name])}</td>
    <td style="width:100%">
      <div class="progress">
        <div class="progress-fill" style="width:${width}%;--progress-color:var(--sweep-${name})"></div>
      </div>
    </td>
    <td class="table-value" data-sign="${escapeHtml(sign)}">${escapeHtml(`${median >= 0 ? "+" : "−"}${Math.abs(median).toFixed(3)}`)}</td>
    <td class="table-value">${escapeHtml(`${Math.round(arm.share_positive * 100)}%`)}</td>
  </tr>`;
}

function runSweep() {
  return withLoading(() => runSweepImpl());
}

async function runSweepImpl() {
  showCompare(true);
  compareGoal.innerHTML = `<strong>20 sampled instructions</strong> &mdash; what each method does across the grammar, not on one phrase`;
  compareColumns.innerHTML = `<div class="card compare-arm" style="grid-column:1/-1">
    <div class="skeleton" style="height:1rem;width:14rem"></div>
    <div class="skeleton" style="height:0.8rem;width:100%"></div>
    <div class="skeleton" style="height:0.8rem;width:100%"></div>
    <div class="skeleton" style="height:0.8rem;width:100%"></div>
  </div>`;

  let payload;
  try {
    const res = await fetch("/api/compare/sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes: 20, seed: 0 }),
    });
    payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "sweep failed");
  } catch (err) {
    compareColumns.innerHTML = `<p class="compare-unavailable">${escapeHtml(err.message)}</p>`;
    return;
  }

  const best = Math.max(...SWEEP_ARMS.map((n) => {
    const median = payload.arms[n]?.median;
    return Number.isFinite(median) ? median : 0;
  }), 0.001);
  compareGoal.innerHTML =
    `<strong>${escapeHtml(payload.episodes)} sampled instructions</strong> on ${escapeHtml(payload.volume)} &mdash; median attainment and how often each method improved on doing nothing`;
  compareColumns.innerHTML = `<div class="card compare-arm" style="grid-column:1/-1">
    <table class="table sweep-table">
      <tr><td></td><td></td><td class="table-value">median</td><td class="table-value">improved</td></tr>
      ${SWEEP_ARMS.map((n) => sweepRow(n, payload.arms[n], best)).join("")}
    </table>
    <p class="compare-unavailable">Search at 200 evaluations is left out here: it costs about a minute over 20 instructions. Use compare for a single instruction to see it.</p>
  </div>`;
}

el("sweep-btn").addEventListener("click", runSweep);
