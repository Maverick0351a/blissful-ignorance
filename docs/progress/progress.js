"use strict";
const $ = id => document.getElementById(id);
// Only show the return link when opened from the running game's own navigation.
$("world-link").hidden = !(/^https?:$/.test(location.protocol) && new URLSearchParams(location.search).get("from") === "world");
const arms = ["starting", "after-16", "after-64"];
const labels = ["Starting", "+16 lives", "+64 lives"];
const percentage = row => 100 * row.prompt_successes / row.lives;

function renderScores() {
  const task = $("task").value, scope = $("scope").value;
  const fixture = window.PROGRESS.fixtures[task];
  const group = scope === "all" ? fixture.arms : fixture.candidates[scope];
  $("task-description").textContent = task === "adjacent"
    ? "Gather and eat ordinary food before tick 64 and before zero fullness."
    : "Eat carried ordinary fruit before tick 16. Separate diagnostic; no gathering required.";
  $("bars").replaceChildren();
  for (const [index, arm] of arms.entries()) {
    const row = group[arm], container = document.createElement("div");
    container.className = "bar-row";
    const label = document.createElement("span"); label.textContent = labels[index];
    const track = document.createElement("div"); track.className = "track"; track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = percentage(row) + "%"; track.append(fill);
    const number = document.createElement("span"); number.className = "bar-number";
    number.textContent = `${row.prompt_successes}/${row.lives} · ${percentage(row).toFixed(1)}%`;
    container.append(label, track, number); $("bars").append(container);
  }
  $("sample-size").textContent = scope === "all"
    ? "96 evaluation lives per condition · six brains · three historical training groups."
    : `16 evaluation lives per condition for ${scope}.`;
}

function tableRow(cells) {
  const row = document.createElement("tr");
  cells.forEach((value, index) => {const cell = document.createElement(index ? "td" : "th");
    if (!index) cell.scope = "row"; cell.textContent = value; row.append(cell);});
  return row;
}
for (const [cid, scores] of Object.entries(window.PROGRESS.fixtures.adjacent.candidates)) {
  $("learner-rows").append(tableRow([cid, ...arms.map(arm => `${scores[arm].prompt_successes}/16`)]));
}
arms.forEach((arm, index) => {
  const row = window.PROGRESS.fixtures.adjacent.arms[arm];
  $("activity-rows").append(tableRow([labels[index], row.ordinary_meals, row.zero_food_ticks,
    row.actions.plant || 0, row.actions.tone || 0]));
});
$("task").addEventListener("change", renderScores); $("scope").addEventListener("change", renderScores); renderScores();

let timer = null;
function stop() {if (timer !== null) clearInterval(timer); timer = null; $("play").textContent = "Play"; $("play").setAttribute("aria-pressed", "false");}
function actionText(action) {
  if (action.verb === "tone") return `tone ${action.tone}`;
  if (action.verb === "move") return `move ${action.direction}`;
  if (action.verb === "make_seeds") return "make seeds";
  return action.verb.replaceAll("_", " ");
}
function drawRoom(rid, episode, frame) {
  const canvas = $("room-" + rid), ctx = canvas.getContext("2d"), b = episode.bounds[rid], a = frame.bodies[rid];
  const cell = canvas.width / (b[2] - b[0] + 1);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  for (let y = b[1]; y <= b[3]; y++) for (let x = b[0]; x <= b[2]; x++) {
    const px = (x - b[0]) * cell, py = (y - b[1]) * cell;
    const grass = episode.terrain[y][x] === 0;
    ctx.fillStyle = grass ? ((x + y) % 2 ? "#5a7850" : "#608057") : "#2f4938";
    ctx.fillRect(px, py, cell, cell);
    if (grass) {ctx.fillStyle = "#6d895b"; ctx.fillRect(px + cell * .18, py + cell * .67, cell * .09, cell * .1);}
    const resource = frame.resources[x + "," + y];
    if (resource) {
      ctx.fillStyle = "#39583b"; ctx.fillRect(px + cell * .23, py + cell * .30, cell * .56, cell * .48);
      ctx.fillStyle = resource.kind === "berry" ? "#dd94a7" : resource.kind === "crop" ? "#a7d565" : "#d9b574";
      ctx.fillRect(px + cell * .29, py + cell * .28, cell * .17, cell * .17);
      ctx.fillRect(px + cell * .54, py + cell * .38, cell * .17, cell * .17);
      ctx.fillRect(px + cell * .37, py + cell * .58, cell * .17, cell * .17);
    }
    if (frame.structures[x + "," + y]) {ctx.fillStyle = "#b39c73";ctx.fillRect(px + 3, py + 3, cell - 6, cell - 6);}
  }
  const x = (a.x - b[0] + .5) * cell, y = (a.y - b[1] + .5) * cell, s = cell / 12;
  ctx.fillStyle = "#28483599"; ctx.fillRect(x - 4 * s, y + 3 * s, 8 * s, 2 * s);
  ctx.fillStyle = "#253d2e"; ctx.fillRect(x - 4 * s, y - 5 * s, 8 * s, 9 * s);
  ctx.fillStyle = rid === "r0" ? "#efd087" : "#b4dce3";
  ctx.fillRect(x - 3 * s, y - 6 * s, 2 * s, 3 * s); ctx.fillRect(x + s, y - 6 * s, 2 * s, 3 * s);
  ctx.fillRect(x - 3 * s, y - 4 * s, 6 * s, 7 * s);
  ctx.fillStyle = "#2e4632"; ctx.fillRect(x - 2 * s, y - 2 * s, s, s);ctx.fillRect(x + s, y - 2 * s, s, s);
  const info = `Fullness ${a.food.toFixed(1)} · pack: ${a.inventory.food} food, ${a.inventory.seed} seeds · ${actionText(a.action)}`;
  $("state-" + rid).textContent = info;
  canvas.setAttribute("aria-label", `${rid === "r0" ? "c0" : "c1"} at tick ${frame.tick}. ${info}`);
  const m = episode.metrics[rid];
  const prompt = m.first_meal_tick !== null && m.first_meal_tick < 64 && m.first_gather_food_tick !== null
    && m.first_gather_food_tick < m.first_meal_tick && (m.first_zero_tick === null || m.first_meal_tick < m.first_zero_tick);
  $("result-" + rid).textContent = `Whole-trial outcome: ${prompt ? "prompt feeding met" : "prompt feeding missed"} · first meal ${m.first_meal_tick === null ? "none" : "tick " + m.first_meal_tick} · ${m.safe_eaten} ordinary meals.`;
}
function renderReplay() {
  const episode = window.RECORDED[$("condition").value];
  $("seek").max = episode.frames.length - 1;
  const frame = episode.frames[Math.min(Number($("seek").value), episode.frames.length - 1)];
  $("tick").textContent = frame.tick + " / " + episode.frames.at(-1).tick;
  $("seek").setAttribute("aria-valuetext", `Recorded tick ${frame.tick} of 512`);
  drawRoom("r0", episode, frame); drawRoom("r1", episode, frame);
}
$("condition").addEventListener("change", () => {stop(); $("seek").value = 0; renderReplay();});
$("seek").addEventListener("input", () => {stop(); renderReplay();});
$("restart").addEventListener("click", () => {stop(); $("seek").value = 0; renderReplay();});
$("play").addEventListener("click", () => {
  if (timer !== null) {stop();return;}
  if (Number($("seek").value) >= Number($("seek").max)) $("seek").value = 0;
  $("play").textContent = "Pause"; $("play").setAttribute("aria-pressed", "true");
  timer = setInterval(() => {$("seek").value = Number($("seek").value) + 1; renderReplay();
    if (Number($("seek").value) >= Number($("seek").max)) stop();}, 160);
});
document.addEventListener("visibilitychange", () => {if (document.hidden) stop();});
stop(); renderReplay();
