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
    setSelectValue(data.dataset);
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
    showToast(`Could not parse that: ${err.detail}`, "destructive");
    return;
  }
  const data = await r.json();
  await refresh(data);
  if (!data.pending) appendMessage(data.current);
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

// ---------- toolbar: toggle groups + single toggle ----------

// A ToggleGroup is a single-select set of buttons sharing one `data-toggle-group`
// container: exactly one child carries data-state="on" at a time, clicking a
// sibling flips it, and the container's mapped `config` key is kept in sync.
function initToggleGroup(container) {
  const key = container.dataset.toggleGroup;
  const items = Array.from(container.querySelectorAll(".toggle-item"));
  items.forEach((item) => {
    item.setAttribute("role", "radio");
    item.setAttribute("aria-checked", item.dataset.state === "on" ? "true" : "false");
    item.addEventListener("click", () => {
      if (item.dataset.state === "on") return;
      items.forEach((other) => {
        other.dataset.state = "off";
        other.setAttribute("aria-checked", "false");
      });
      item.dataset.state = "on";
      item.setAttribute("aria-checked", "true");
      config[key] = item.dataset.value;
    });
  });
}

document.querySelectorAll("[data-toggle-group]").forEach(initToggleGroup);

const searchToggleBtn = el("search-toggle-btn");
const searchOptions = el("search-options");
searchToggleBtn.addEventListener("click", () => {
  config.search = !config.search;
  searchToggleBtn.dataset.active = String(config.search);
  searchOptions.hidden = !config.search;
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

async function chooseDataset(name) {
  if (name === state.dataset) { closeSelect(); return; }
  const previous = state.dataset;
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
  closeSelect();
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  await refresh(data);
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
  // Dialog starts at opacity/scale 0 in CSS; adding this class one frame
  // later is what makes the transition to full opacity/scale actually run
  // (toggling it in the same frame as showModal() would skip straight to
  // the end state with no visible animation).
  requestAnimationFrame(() => modal.classList.add("dialog-open"));
}

function closeCommandsModal() {
  const modal = el("commands-modal");
  modal.classList.remove("dialog-open");
  modal.close();
}

el("commands-help-btn").addEventListener("click", openCommandsModal);
el("commands-modal-close").addEventListener("click", closeCommandsModal);
el("commands-modal").addEventListener("click", (e) => {
  if (e.target === el("commands-modal")) closeCommandsModal();
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
