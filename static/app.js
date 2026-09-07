const state = { current: null, cursor: 0, total: 1, pending: null, dataset: null };
const config = { parser: "rule", search: false, evaluator: "objective" };

const THUMB_UP_PATH = "M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3";
const THUMB_DOWN_PATH = "M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17";

const el = (id) => document.getElementById(id);
const singleView = el("single-view");
const judgeView = el("judge-view");
const messagesEl = el("messages");
const emptyState = el("empty-state");
const textInput = el("text-input");
const sendBtn = el("send-btn");

async function refresh(data) {
  state.current = data.current;
  state.cursor = data.cursor;
  state.total = data.total;
  state.pending = data.pending;
  if (data.dataset && data.dataset !== state.dataset) {
    state.dataset = data.dataset;
    const select = el("dataset-select");
    if (select.value !== data.dataset) select.value = data.dataset;
  }

  el("step-counter").textContent = `${state.cursor + 1} / ${state.total}`;
  el("back-btn").disabled = state.cursor <= 0 || !!state.pending;
  el("forward-btn").disabled = state.cursor >= state.total - 1 || !!state.pending;
  el("reset-btn").disabled = !!state.pending;

  if (state.pending) {
    singleView.hidden = true;
    judgeView.hidden = false;
    el("before-image").src = `data:image/png;base64,${state.pending.before_image_b64}`;
    el("after-image").src = `data:image/png;base64,${state.pending.after_image_b64}`;
    el("judge-progress").textContent = `search step ${state.pending.iteration + 1} / ${state.pending.max_steps}`;
  } else {
    singleView.hidden = false;
    judgeView.hidden = true;
    el("current-image").src = `data:image/png;base64,${state.current.image_b64}`;
  }

  updateTelemetry(state.current.masses);
}

function updateTelemetry(masses) {
  if (!masses) return;
  for (const tissue of ["air", "fat", "soft", "spongy", "bone"]) {
    const v = masses[tissue];
    el(`telem-${tissue}`).textContent = v === undefined ? "—" : v.toFixed(1);
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

  const tags = document.createElement("div");
  tags.className = "msg-tags";
  if (step.search) {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = "search";
    tags.appendChild(t);
  }
  if (step.verdict === 1 || step.verdict === -1) {
    const t = document.createElement("span");
    t.className = `tag ${step.verdict === 1 ? "good" : "bad"}`;
    t.textContent = step.verdict === 1 ? "better" : "worse";
    tags.appendChild(t);
  }
  if (tags.children.length) div.appendChild(tags);

  const img = document.createElement("img");
  img.src = `data:image/png;base64,${step.image_b64}`;
  div.appendChild(img);

  div.appendChild(buildFeedbackRow(step));

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function buildFeedbackRow(step) {
  const row = document.createElement("div");
  row.className = "msg-feedback";

  const makeBtn = (rating, path) => {
    const btn = document.createElement("button");
    btn.className = "feedback-btn";
    btn.setAttribute("aria-label", rating === "up" ? "Good result" : "Bad result");
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg>`;
    if (step.feedback === rating) btn.classList.add("active", rating);
    btn.addEventListener("click", async () => {
      const r = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step_id: step.id, rating }),
      });
      if (r.status !== 200) return;
      step.feedback = rating;
      row.querySelectorAll(".feedback-btn").forEach((b) => b.classList.remove("active", "up", "down"));
      btn.classList.add("active", rating);
    });
    return btn;
  };

  row.appendChild(makeBtn("up", THUMB_UP_PATH));
  row.appendChild(makeBtn("down", THUMB_DOWN_PATH));
  return row;
}

async function loadState() {
  const r = await fetch("/api/state");
  const data = await r.json();
  await refresh(data);
  // Full history isn't in /api/state (only the current step), so rebuild the
  // thread by walking back/forward would be wasteful; instead the server's
  // current step is enough to seed the empty state on first load.
  if (data.current.cmd_text) {
    messagesEl.innerHTML = "";
    appendMessage(data.current);
  }
}

async function sendCommand(text) {
  const body = {
    text,
    parser: config.parser,
    search: config.search,
    evaluator: config.evaluator,
    steps: parseInt(el("steps-input").value, 10),
  };
  const r = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.status !== 200) {
    const err = await r.json();
    appendError(err.detail);
    return;
  }
  const data = await r.json();
  await refresh(data);
  if (!data.pending) appendMessage(data.current);
}

function appendError(detail) {
  if (emptyState.parentNode === messagesEl) messagesEl.removeChild(emptyState);
  const div = document.createElement("div");
  div.className = "msg";
  div.style.color = "var(--bad)";
  div.textContent = `Could not parse that: ${detail}`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function judge(verdict) {
  const r = await fetch("/api/judge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ verdict }),
  });
  const data = await r.json();
  await refresh(data);
  if (!data.pending) appendMessage(data.current);
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

el("back-btn").addEventListener("click", async () => refresh(await (await fetch("/api/back", { method: "POST" })).json()));
el("forward-btn").addEventListener("click", async () => refresh(await (await fetch("/api/forward", { method: "POST" })).json()));
el("reset-btn").addEventListener("click", () => sendCommand("reset"));
el("better-btn").addEventListener("click", () => judge("better"));
el("worse-btn").addEventListener("click", () => judge("worse"));

// ---------- toolbar chips ----------

const parserToggle = el("parser-toggle");
parserToggle.addEventListener("click", () => {
  config.parser = config.parser === "rule" ? "llm" : "rule";
  parserToggle.textContent = `${config.parser} parser`;
  parserToggle.dataset.active = config.parser === "llm" ? "true" : "false";
});

const searchToggleBtn = el("search-toggle-btn");
const searchOptions = el("search-options");
searchToggleBtn.addEventListener("click", () => {
  config.search = !config.search;
  searchToggleBtn.dataset.active = String(config.search);
  searchOptions.hidden = !config.search;
});

const evaluatorToggle = el("evaluator-toggle");
evaluatorToggle.addEventListener("click", () => {
  config.evaluator = config.evaluator === "objective" ? "human" : "objective";
  evaluatorToggle.textContent = config.evaluator;
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

micBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    appendError("microphone access denied or unavailable");
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
      const r = await fetch("/api/transcribe", { method: "POST", body: form });
      const data = await r.json();
      if (data.text) sendCommand(data.text);
    } finally {
      micBtn.disabled = false;
    }
  };
  mediaRecorder.start();
  micBtn.classList.add("recording");
  micIcon.style.display = "none";
  micWaveform.hidden = false;
  startWaveform(stream);
});

// ---------- dataset selector ----------

const datasetSelect = el("dataset-select");

async function loadDatasets() {
  const r = await fetch("/api/datasets");
  const data = await r.json();
  datasetSelect.innerHTML = "";
  data.available.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    datasetSelect.appendChild(opt);
  });
  datasetSelect.value = data.current;
  state.dataset = data.current;
}

datasetSelect.addEventListener("change", async () => {
  const r = await fetch("/api/dataset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: datasetSelect.value }),
  });
  if (r.status !== 200) {
    const err = await r.json();
    alert(`Could not switch dataset: ${err.detail}`);
    datasetSelect.value = state.dataset;
    return;
  }
  const data = await r.json();
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  await refresh(data);
});

// ---------- command reference modal ----------

let commandsCache = null;

async function openCommandsModal() {
  const modal = el("commands-modal");
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
}

el("commands-help-btn").addEventListener("click", openCommandsModal);
el("commands-modal-close").addEventListener("click", () => el("commands-modal").close());
el("commands-modal").addEventListener("click", (e) => {
  if (e.target === el("commands-modal")) el("commands-modal").close();
});

// ---------- manual camera rotation (drag) + zoom (scroll) ----------

const DRAG_DEGREES_PER_PIXEL = 0.4;
const DRAG_THROTTLE_MS = 110;
const WHEEL_ZOOM_SENSITIVITY = 0.0015;
const WHEEL_COMMIT_DEBOUNCE_MS = 300;

const currentImage = el("current-image");

async function sendCameraDelta(dAzimuth, dElevation, dZoomFactor, commit) {
  const r = await fetch("/api/camera_delta", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ d_azimuth: dAzimuth, d_elevation: dElevation, d_zoom_factor: dZoomFactor, commit }),
  });
  return r.json();
}

let dragging = false;
let dragStartX = 0;
let dragStartY = 0;
let lastDragSendAt = 0;

currentImage.addEventListener("pointerdown", (e) => {
  if (state.pending) return;
  dragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  lastDragSendAt = 0;
  currentImage.setPointerCapture(e.pointerId);
  currentImage.classList.add("dragging");
});

currentImage.addEventListener("pointermove", async (e) => {
  if (!dragging) return;
  const now = performance.now();
  if (now - lastDragSendAt < DRAG_THROTTLE_MS) return;
  lastDragSendAt = now;
  const dAzimuth = (e.clientX - dragStartX) * DRAG_DEGREES_PER_PIXEL;
  const dElevation = -(e.clientY - dragStartY) * DRAG_DEGREES_PER_PIXEL;
  const data = await sendCameraDelta(dAzimuth, dElevation, 1.0, false);
  currentImage.src = `data:image/png;base64,${data.image_b64}`;
});

currentImage.addEventListener("pointerup", async (e) => {
  if (!dragging) return;
  dragging = false;
  currentImage.classList.remove("dragging");
  const dAzimuth = (e.clientX - dragStartX) * DRAG_DEGREES_PER_PIXEL;
  const dElevation = -(e.clientY - dragStartY) * DRAG_DEGREES_PER_PIXEL;
  const data = await sendCameraDelta(dAzimuth, dElevation, 1.0, true);
  await refresh(data);
  appendMessage(data.current);
});

let wheelZoomAccum = 1.0;
let wheelCommitTimer = null;

currentImage.addEventListener("wheel", async (e) => {
  if (state.pending) return;
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * WHEEL_ZOOM_SENSITIVITY);
  wheelZoomAccum *= factor;
  const data = await sendCameraDelta(0.0, 0.0, wheelZoomAccum, false);
  currentImage.src = `data:image/png;base64,${data.image_b64}`;

  if (wheelCommitTimer) clearTimeout(wheelCommitTimer);
  wheelCommitTimer = setTimeout(async () => {
    const finalZoom = wheelZoomAccum;
    wheelZoomAccum = 1.0;
    const finalData = await sendCameraDelta(0.0, 0.0, finalZoom, true);
    await refresh(finalData);
    appendMessage(finalData.current);
  }, WHEEL_COMMIT_DEBOUNCE_MS);
}, { passive: false });

updateSendState();
loadDatasets();
loadState();
