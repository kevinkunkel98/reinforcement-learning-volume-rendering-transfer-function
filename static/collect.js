/* /collect: judge pairs of rendered results, one at a time.
 *
 * Each item's images arrive already embedded as data URLs in the /next and
 * /judge responses (see collect.py), so there is no separate image fetch --
 * "decision time" is measured from when the two candidate <img> elements
 * finish loading (decoding a data URL still fires a load event
 * asynchronously) to the moment a choice is made.
 */
const RATER_KEY = "collectRaterId";
const ROLE_KEY = "collectRaterRole";
const EXPERIENCE_KEY = "collectRaterExperience";
const VALID_ROLES = ["radiologist", "clinician", "researcher", "other"];

function getRaterId() {
  let id = localStorage.getItem(RATER_KEY);
  if (!id) {
    id = (window.prompt("Rater ID:") || "anon").trim() || "anon";
    localStorage.setItem(RATER_KEY, id);
  }
  return id;
}

// Asked once per browser (stored in localStorage, same as the rater ID) --
// role is validated against VALID_ROLES so /api/collect/next always gets one
// of the four values the server accepts; experience is free text and may be
// left blank.
function getRaterRole() {
  let role = localStorage.getItem(ROLE_KEY);
  while (!role || !VALID_ROLES.includes(role)) {
    role = (window.prompt(`Role (${VALID_ROLES.join("/")}):`) || "").trim().toLowerCase();
  }
  localStorage.setItem(ROLE_KEY, role);
  return role;
}

function getRaterExperience() {
  let experience = localStorage.getItem(EXPERIENCE_KEY);
  if (experience === null) {
    experience = (window.prompt("Experience (optional, e.g. \"8 years CT\"):") || "").trim();
    localStorage.setItem(EXPERIENCE_KEY, experience);
  }
  return experience;
}

const raterId = getRaterId();
const raterRole = getRaterRole();
const raterExperience = getRaterExperience();
const countKey = `collectJudgedCount:${raterId}`;

function loadJudgedCount() {
  const stored = Number.parseInt(localStorage.getItem(countKey) || "0", 10);
  return Number.isFinite(stored) && stored >= 0 ? stored : 0;
}

let judgedCount = loadJudgedCount();

const el = (id) => document.getElementById(id);
const instructionEl = el("collect-instruction");
const volumeEl = el("collect-volume");
const progressEl = el("collect-progress");
const raterEl = el("collect-rater");
const aImage = el("candidate-a-image");
const bImage = el("candidate-b-image");
const startImage = el("start-image");
const startPanel = el("start-panel");
const showStartToggle = el("show-start-toggle");
const judgeButtons = Array.from(document.querySelectorAll(".judge-btn"));

let currentItem = null;
let decisionStart = null;
let busy = false;

raterEl.textContent = `rater: ${raterId}`;
progressEl.textContent = `judged: ${judgedCount}`;

function setBusy(value) {
  busy = value;
  for (const btn of judgeButtons) btn.disabled = value;
}

function loadImage(imgEl, src) {
  return new Promise((resolve) => {
    imgEl.onload = () => resolve();
    imgEl.onerror = () => resolve();
    imgEl.src = src;
  });
}

async function showItem(item) {
  currentItem = item;
  decisionStart = null;
  instructionEl.textContent = item.text;
  volumeEl.textContent = item.volume;

  await Promise.all([
    loadImage(aImage, item.a_image),
    loadImage(bImage, item.b_image),
  ]);
  startImage.src = item.start_image;
  decisionStart = performance.now();
  setBusy(false);
}

async function fetchNext() {
  const response = await fetch("/api/collect/next", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rater_id: raterId, rater_role: raterRole, rater_experience: raterExperience }),
  });
  if (!response.ok) throw new Error(`next failed: ${response.status}`);
  return response.json();
}

async function judge(choice) {
  if (busy || !currentItem) return;
  setBusy(true);
  const decisionMs = decisionStart != null ? Math.round(performance.now() - decisionStart) : null;
  const pairId = currentItem.pair_id;
  try {
    const response = await fetch("/api/collect/judge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pair_id: pairId, choice, decision_ms: decisionMs }),
    });
    if (!response.ok) throw new Error(`judge failed: ${response.status}`);
    const next = await response.json();
    judgedCount += 1;
    localStorage.setItem(countKey, String(judgedCount));
    progressEl.textContent = `judged: ${judgedCount}`;
    await showItem(next);
  } catch (err) {
    console.error(err);
    setBusy(false);
  }
}

for (const btn of judgeButtons) {
  btn.addEventListener("click", () => judge(btn.dataset.choice));
}

showStartToggle.addEventListener("click", () => {
  startPanel.hidden = !startPanel.hidden;
});

const KEY_CHOICES = { a: "a", b: "b", e: "equal", s: "skip" };

document.addEventListener("keydown", (event) => {
  const active = document.activeElement;
  if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
  const choice = KEY_CHOICES[event.key.toLowerCase()];
  if (choice) {
    event.preventDefault();
    judge(choice);
  }
});

fetchNext().then(showItem).catch((err) => {
  console.error(err);
  instructionEl.textContent = "Failed to load item.";
});
