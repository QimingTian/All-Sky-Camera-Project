const state = {
  project: null,
  selectedBase: null,
  index: 0,
  imageMeta: { width: 1920, height: 1080, scale: 1 },
  img: new Image(),
};

const twilightGrid = document.getElementById("twilight-grid");
const toStep2 = document.getElementById("to-step-2");
const targetTime = document.getElementById("target-time");
const sunRadius = document.getElementById("sun-radius");
const sunRadiusVal = document.getElementById("sun-radius-val");
const applyTime = document.getElementById("apply-time");
const canvas = document.getElementById("frame-canvas");
const ctx = canvas.getContext("2d");
const frameLabel = document.getElementById("frame-label");
const progressText = document.getElementById("progress-text");
const progressBar = document.getElementById("progress-bar");
const baseThumb = document.getElementById("base-thumb");
const renderPreview = document.getElementById("render-preview");
const renderResult = document.getElementById("render-result");

function showStep(n) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
  document.getElementById(`step-${n}`).classList.add("active");
  document.querySelectorAll(".step-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.step === String(n));
  });
}

document.querySelectorAll(".step-tab").forEach((btn) => {
  btn.addEventListener("click", () => showStep(Number(btn.dataset.step)));
});

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function timeToApi(value) {
  return value.replace(":", "");
}

function firstUndoneIndex(frames) {
  const i = frames.findIndex((f) => !f.done);
  return i === -1 ? Math.max(0, frames.length - 1) : i;
}

async function loadTwilightCandidates() {
  twilightGrid.innerHTML = "<p>加载底片选项…</p>";
  const items = await api("/api/twilight-candidates");
  twilightGrid.innerHTML = "";
  if (!items.length) {
    twilightGrid.innerHTML = "<p>该时段暂无帧。</p>";
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "card";
    card.dataset.filename = item.filename;
    card.innerHTML = `
      <img src="/api/image/${item.filename}?w=480" alt="${item.filename}" loading="lazy" />
      <div class="meta">
        <strong>${item.date} · ${item.timeLabel}</strong>
      </div>`;
    card.addEventListener("click", () => {
      document.querySelectorAll(".card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      state.selectedBase = item.filename;
      toStep2.disabled = false;
    });
    twilightGrid.appendChild(card);
  });
}

async function setupProject() {
  const project = await api("/api/project/setup", {
    method: "POST",
    body: JSON.stringify({
      basePlate: state.selectedBase,
      targetTime: timeToApi(targetTime.value),
      sunRadius: Number(sunRadius.value),
    }),
  });
  state.project = project;
  state.index = firstUndoneIndex(project.frames);
  baseThumb.src = `/api/image/${project.basePlate}?w=480`;
  updateProgress();
  await loadCurrentFrame();
}

async function refreshProject() {
  state.project = await api("/api/project");
  if (state.project.basePlate) {
    state.selectedBase = state.project.basePlate;
    baseThumb.src = `/api/image/${state.project.basePlate}?w=480`;
    targetTime.value = `${state.project.targetTime.slice(0, 2)}:${state.project.targetTime.slice(2)}`;
    sunRadius.value = state.project.sunRadius || 6;
    sunRadiusVal.textContent = sunRadius.value;
    state.index = firstUndoneIndex(state.project.frames);
    updateProgress();
    await loadCurrentFrame();
  }
}

function currentFrame() {
  return state.project?.frames?.[state.index] || null;
}

function updateProgress() {
  const total = state.project?.frames?.length || 0;
  const frames = state.project?.frames || [];
  const done = frames.filter((f) => f.done).length;
  const suns = frames.filter((f) => f.done && !f.skipped).length;
  progressText.textContent = `${done} / ${total} 已处理 · ${suns} 个太阳`;
  progressBar.style.width = total ? `${(done / total) * 100}%` : "0%";
}

function drawOverlay() {
  const frame = currentFrame();
  if (!state.img.complete || !frame) return;
  const scale = state.imageMeta.scale;
  canvas.width = state.img.naturalWidth;
  canvas.height = state.img.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0);

  if (frame.predX != null && frame.predY != null) {
    const px = frame.predX * scale;
    const py = frame.predY * scale;
    ctx.strokeStyle = "rgba(0, 230, 255, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px - 10, py);
    ctx.lineTo(px + 10, py);
    ctx.moveTo(px, py - 10);
    ctx.lineTo(px, py + 10);
    ctx.stroke();
  }

  if (frame.x == null || frame.y == null) return;
  const x = frame.x * scale;
  const y = frame.y * scale;
  const r = Math.max(Number(sunRadius.value) * scale, 4);
  ctx.setLineDash(frame.skipped ? [5, 4] : []);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.strokeStyle = frame.skipped ? "rgba(255, 80, 80, 0.95)" : "rgba(255, 179, 71, 0.95)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(x, y, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = frame.skipped ? "#ff5050" : "#ffb347";
  ctx.fill();
}

function frameStatus(frame) {
  if (!frame.done) return "未处理";
  const who = frame.source === "manual" ? "手动" : "自动";
  if (frame.skipped) return `${who}跳过${frame.reason ? `（${frame.reason}）` : ""}`;
  return `${who}定位`;
}

async function loadCurrentFrame() {
  const frame = currentFrame();
  if (!frame) {
    frameLabel.textContent = "没有该时刻的帧";
    return;
  }
  frameLabel.textContent = `${state.index + 1}/${state.project.frames.length} · ${frame.date} · ${frame.timeLabel} · ${frame.file} · ${frameStatus(frame)}`;
  state.imageMeta = await api(`/api/image-meta/${frame.file}?w=1280`);
  await new Promise((resolve, reject) => {
    state.img.onload = resolve;
    state.img.onerror = reject;
    state.img.src = `/api/image/${frame.file}?w=1280&ts=${Date.now()}`;
  });
  drawOverlay();
}

function canvasCoords(evt) {
  const rect = canvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) * (canvas.width / rect.width);
  const y = (evt.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

async function submitAnnotation(payload) {
  const data = await api("/api/project/annotate", {
    method: "POST",
    body: JSON.stringify({ index: state.index, ...payload }),
  });
  state.project.frames[state.index] = data.frame;
  state.project.doneCount = data.doneCount;
  updateProgress();
  if (state.index < state.project.frames.length - 1) {
    state.index += 1;
    await loadCurrentFrame();
  }
}

canvas.addEventListener("click", async (evt) => {
  if (!state.project?.frames?.length) return;
  const { x, y } = canvasCoords(evt);
  const scale = state.imageMeta.scale;
  await submitAnnotation({
    x: x / scale,
    y: y / scale,
    skipped: false,
  });
});

document.getElementById("btn-skip").addEventListener("click", async () => {
  if (!state.project?.frames?.length) return;
  await submitAnnotation({ skipped: true });
});

async function goTo(index) {
  const n = state.project?.frames?.length || 0;
  if (!n) return;
  state.index = Math.min(Math.max(index, 0), n - 1);
  await loadCurrentFrame();
}

document.getElementById("btn-prev").addEventListener("click", () => goTo(state.index - 1));
document.getElementById("btn-next").addEventListener("click", () => goTo(state.index + 1));

document.getElementById("btn-back").addEventListener("click", async () => {
  if (!state.project?.frames?.length) return;
  await api("/api/project/reset-frame", {
    method: "POST",
    body: JSON.stringify({ index: state.index }),
  });
  state.project = await api("/api/project");
  updateProgress();
  await loadCurrentFrame();
});

document.getElementById("btn-autodetect").addEventListener("click", async () => {
  if (!state.project?.frames?.length) {
    alert("请先选择底片并应用时刻");
    return;
  }
  const out = document.getElementById("autodetect-result");
  out.textContent = "检测中…（每张约 0.1 s）";
  try {
    const data = await api("/api/project/autodetect", { method: "POST", body: "{}" });
    state.project = data;
    const s = data.stats;
    out.textContent = `定位 ${s.detected} 天 · 云中误检剔除 ${s.rejected} · 无清晰太阳 ${s.none}（手动标注的不会被覆盖）`;
    updateProgress();
    state.index = 0;
    await loadCurrentFrame();
  } catch (err) {
    out.textContent = err.message;
  }
});

document.addEventListener("keydown", (evt) => {
  if (evt.target.tagName === "INPUT") return;
  if (evt.key === "s" || evt.key === "S") document.getElementById("btn-skip").click();
  if (evt.key === "ArrowLeft") document.getElementById("btn-prev").click();
  if (evt.key === "ArrowRight") document.getElementById("btn-next").click();
});

sunRadius.addEventListener("input", () => {
  sunRadiusVal.textContent = sunRadius.value;
  drawOverlay();
});

toStep2.addEventListener("click", async () => {
  if (!state.selectedBase) return;
  await setupProject();
  showStep(2);
});

applyTime.addEventListener("click", async () => {
  if (!state.selectedBase && !state.project?.basePlate) {
    alert("请先选择底片");
    showStep(1);
    return;
  }
  state.selectedBase = state.selectedBase || state.project.basePlate;
  await setupProject();
});

document.getElementById("to-step-3").addEventListener("click", () => showStep(3));
document.getElementById("back-annotate").addEventListener("click", () => showStep(2));

document.getElementById("btn-render").addEventListener("click", async () => {
  renderResult.textContent = "合成中…";
  try {
    const data = await api("/api/render", {
      method: "POST",
      body: JSON.stringify({ sunRadius: Number(sunRadius.value) }),
    });
    renderPreview.src = `${data.previewUrl}?ts=${Date.now()}`;
    renderPreview.style.display = "block";
    renderResult.textContent = `完成：${data.count} 颗太阳\n${data.jpg} / ${data.tiff}\n保存在 Half Hour/analemma/output/`;
  } catch (err) {
    renderResult.textContent = err.message;
  }
});

(async function init() {
  await loadTwilightCandidates();
  await refreshProject();
  if (state.project?.basePlate) {
    state.selectedBase = state.project.basePlate;
    toStep2.disabled = false;
    document.querySelectorAll(".card").forEach((card) => {
      if (card.dataset.filename === state.project.basePlate) {
        card.classList.add("selected");
      }
    });
  }
})();
