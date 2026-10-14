const IMAGE_SRC = 'image.png';
const IMAGE_NAME = '2026-09-15_22-26-55_seq0001of0001_gain250_exp60.000s.tiff';

const COMMON_STARS = [
  'Polaris', 'Vega', 'Deneb', 'Altair', 'Arcturus', 'Capella', 'Sirius', 'Procyon',
  'Betelgeuse', 'Rigel', 'Aldebaran', 'Antares', 'Spica', 'Regulus', 'Fomalhaut',
  'Alpheratz', 'Mirach', 'Algol', 'Mirfak', 'Schedar', 'Caph', 'Dubhe', 'Merak',
  'Phecda', 'Megrez', 'Alioth', 'Mizar', 'Alkaid', 'Kochab', 'Eltanin', 'Albireo',
  'Sadr', 'Enif', 'Markab', 'Scheat', 'Alderamin', 'Hamal', 'Menkar', 'Alphecca',
  'Izar', 'Rasalhague', 'Rasalgethi', 'Kaus Australis', 'Nunki', 'Atria',
  'Jupiter', 'Saturn', 'Mars', 'Venus', 'Moon'
];

const view = document.getElementById('view');
const vctx = view.getContext('2d');
const loupe = document.getElementById('loupe');
const lctx = loupe.getContext('2d');
const stage = document.getElementById('stage');
const hud = document.getElementById('hud');

const el = id => document.getElementById(id);
const editor = el('editor');
const nameInput = el('nameInput');

// --- state ---------------------------------------------------------------
let img = null;              // raw image bitmap
let raw = null;              // ImageData of the raw image (for snapping)
let adjusted = null;         // offscreen canvas holding brightness-adjusted pixels
let stars = [];              // [{id, name, x, y}]
let selectedId = null;       // star currently open in the editor
let draftPoint = null;       // {x, y} not yet named
let nextId = 1;

let scale = 1, offX = 0, offY = 0;   // image -> screen transform
let pointer = null;                   // last mouse position in image space
let dragging = null;                  // {mode:'pan'|'point', ...}
let spaceDown = false;

// --- bootstrap -----------------------------------------------------------
el('starNames').innerHTML = COMMON_STARS.map(n => `<option value="${n}">`).join('');

const image = new Image();
image.onload = async () => {
  img = image;
  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  raw = cx.getImageData(0, 0, img.width, img.height);

  adjusted = document.createElement('canvas');
  adjusted.width = img.width;
  adjusted.height = img.height;

  await loadLabels();
  applyAdjustments();
  resize();
  resetView();
};
image.src = IMAGE_SRC;

// --- brightness / gamma / black point ------------------------------------
function applyAdjustments() {
  if (!raw) return;
  const brightness = parseFloat(el('brightness').value);
  const gamma = parseFloat(el('gamma').value);
  const black = parseFloat(el('black').value);

  const lut = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) {
    let v = (i - black) / (255 - black);
    if (v < 0) v = 0;
    v = Math.pow(v * brightness, gamma);
    lut[i] = Math.min(255, v * 255);
  }

  const src = raw.data;
  const out = new ImageData(raw.width, raw.height);
  const dst = out.data;
  for (let i = 0; i < src.length; i += 4) {
    dst[i] = lut[src[i]];
    dst[i + 1] = lut[src[i + 1]];
    dst[i + 2] = lut[src[i + 2]];
    dst[i + 3] = 255;
  }
  adjusted.getContext('2d').putImageData(out, 0, 0);
  draw();
}

for (const [id, fmt] of [
  ['brightness', v => v.toFixed(1) + '×'],
  ['gamma', v => v.toFixed(2)],
  ['black', v => String(Math.round(v))]
]) {
  const input = el(id);
  const label = el(id + 'Val');
  label.textContent = fmt(parseFloat(input.value));
  input.addEventListener('input', () => {
    label.textContent = fmt(parseFloat(input.value));
    applyAdjustments();
  });
}

// --- view transform ------------------------------------------------------
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = stage.getBoundingClientRect();
  view.width = Math.round(r.width * dpr);
  view.height = Math.round(r.height * dpr);
  view.style.width = r.width + 'px';
  view.style.height = r.height + 'px';
  vctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  loupe.width = 180 * dpr;
  loupe.height = 180 * dpr;
  draw();
}
window.addEventListener('resize', resize);

function resetView() {
  if (!img) return;
  const r = stage.getBoundingClientRect();
  scale = Math.min(r.width / img.width, r.height / img.height);
  offX = (r.width - img.width * scale) / 2;
  offY = (r.height - img.height * scale) / 2;
  draw();
}
el('reset').addEventListener('click', resetView);

const toImage = (sx, sy) => ({ x: (sx - offX) / scale, y: (sy - offY) / scale });
const toScreen = (ix, iy) => ({ x: ix * scale + offX, y: iy * scale + offY });

function draw() {
  if (!img || !adjusted) return;
  const r = stage.getBoundingClientRect();
  vctx.clearRect(0, 0, r.width, r.height);
  vctx.imageSmoothingEnabled = scale < 2.5;
  vctx.drawImage(adjusted, offX, offY, img.width * scale, img.height * scale);

  for (const s of stars) drawMarker(s, s.id === selectedId);
  if (draftPoint) drawMarker(draftPoint, true, true);

  if (pointer) {
    hud.textContent = `x ${pointer.x.toFixed(1)}  y ${pointer.y.toFixed(1)}   ${(scale * 100).toFixed(0)}%`;
  } else {
    hud.textContent = `${(scale * 100).toFixed(0)}%`;
  }
}

function drawMarker(p, active, draft = false) {
  const { x, y } = toScreen(p.x, p.y);
  const color = draft ? '#ffd166' : active ? '#6ea8ff' : '#48d597';
  const rad = 9;
  vctx.save();
  vctx.strokeStyle = color;
  vctx.lineWidth = active || draft ? 2 : 1.4;
  vctx.beginPath();
  vctx.arc(x, y, rad, 0, Math.PI * 2);
  vctx.stroke();
  vctx.beginPath();
  vctx.moveTo(x - rad - 5, y); vctx.lineTo(x - rad - 1, y);
  vctx.moveTo(x + rad + 1, y); vctx.lineTo(x + rad + 5, y);
  vctx.moveTo(x, y - rad - 5); vctx.lineTo(x, y - rad - 1);
  vctx.moveTo(x, y + rad + 1); vctx.lineTo(x, y + rad + 5);
  vctx.stroke();
  if (p.name) {
    vctx.font = '600 12px -apple-system, sans-serif';
    const w = vctx.measureText(p.name).width;
    vctx.fillStyle = 'rgba(8,10,14,.75)';
    vctx.fillRect(x + rad + 6, y - 9, w + 8, 17);
    vctx.fillStyle = color;
    vctx.fillText(p.name, x + rad + 10, y + 3.5);
  }
  vctx.restore();
}

// --- star snapping -------------------------------------------------------
function snapToStar(ix, iy, radius = 7) {
  if (!raw) return { x: ix, y: iy };
  const { width, height, data } = raw;
  const cx = Math.round(ix), cy = Math.round(iy);
  const lum = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return 0;
    const i = (y * width + x) * 4;
    return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  };

  let best = -1, bx = cx, by = cy;
  for (let y = cy - radius; y <= cy + radius; y++) {
    for (let x = cx - radius; x <= cx + radius; x++) {
      const v = lum(x, y);
      if (v > best) { best = v; bx = x; by = y; }
    }
  }

  // local background from the ring around the peak, then flux-weighted centroid
  let bgSum = 0, bgN = 0;
  const win = 3;
  for (let y = by - win - 2; y <= by + win + 2; y++) {
    for (let x = bx - win - 2; x <= bx + win + 2; x++) {
      if (Math.abs(x - bx) <= win && Math.abs(y - by) <= win) continue;
      bgSum += lum(x, y); bgN++;
    }
  }
  const bg = bgN ? bgSum / bgN : 0;
  if (best - bg < 6) return { x: ix, y: iy };   // nothing star-like here

  let sw = 0, sx = 0, sy = 0;
  for (let y = by - win; y <= by + win; y++) {
    for (let x = bx - win; x <= bx + win; x++) {
      const w = Math.max(0, lum(x, y) - bg);
      sw += w; sx += w * x; sy += w * y;
    }
  }
  return sw > 0 ? { x: sx / sw, y: sy / sw } : { x: bx, y: by };
}

function hitTest(ix, iy) {
  const tol = 10 / scale;
  let found = null, bestD = Infinity;
  for (const s of stars) {
    const d = Math.hypot(s.x - ix, s.y - iy);
    if (d < tol && d < bestD) { bestD = d; found = s; }
  }
  return found;
}

// --- loupe ---------------------------------------------------------------
function drawLoupe(sx, sy, ip) {
  if (!el('showLoupe').checked || !adjusted) { loupe.style.display = 'none'; return; }
  const dpr = window.devicePixelRatio || 1;
  const size = 180, zoom = 8, src = size / zoom;
  loupe.style.display = 'block';
  const r = stage.getBoundingClientRect();
  loupe.style.left = Math.min(Math.max(sx + 24, 4), r.width - size - 4) + 'px';
  loupe.style.top = Math.min(Math.max(sy - size - 24, 4), r.height - size - 4) + 'px';

  lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  lctx.clearRect(0, 0, size, size);
  lctx.imageSmoothingEnabled = false;
  lctx.drawImage(adjusted, ip.x - src / 2, ip.y - src / 2, src, src, 0, 0, size, size);
  lctx.strokeStyle = 'rgba(255,209,102,.9)';
  lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(size / 2, size / 2 - 14); lctx.lineTo(size / 2, size / 2 - 4);
  lctx.moveTo(size / 2, size / 2 + 4); lctx.lineTo(size / 2, size / 2 + 14);
  lctx.moveTo(size / 2 - 14, size / 2); lctx.lineTo(size / 2 - 4, size / 2);
  lctx.moveTo(size / 2 + 4, size / 2); lctx.lineTo(size / 2 + 14, size / 2);
  lctx.stroke();
}

// --- mouse ---------------------------------------------------------------
view.addEventListener('mousedown', e => {
  const r = stage.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const ip = toImage(sx, sy);

  if (e.button === 2 || e.button === 1 || spaceDown) {
    dragging = { mode: 'pan', sx, sy, offX, offY };
    return;
  }
  const hit = hitTest(ip.x, ip.y);
  if (hit) {
    dragging = { mode: 'point', star: hit, moved: false, dx: hit.x - ip.x, dy: hit.y - ip.y };
    return;
  }
  dragging = { mode: 'new', sx, sy, moved: false };
});

view.addEventListener('mousemove', e => {
  const r = stage.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  pointer = toImage(sx, sy);

  if (dragging?.mode === 'pan') {
    offX = dragging.offX + (sx - dragging.sx);
    offY = dragging.offY + (sy - dragging.sy);
  } else if (dragging?.mode === 'point') {
    if (Math.abs(sx - (toScreen(dragging.star.x, dragging.star.y).x)) > 0) dragging.moved = true;
    dragging.star.x = pointer.x + dragging.dx;
    dragging.star.y = pointer.y + dragging.dy;
    scheduleSave();
    renderList();
  } else if (dragging?.mode === 'new') {
    if (Math.hypot(sx - dragging.sx, sy - dragging.sy) > 3) dragging.moved = true;
  }

  draw();
  drawLoupe(sx, sy, pointer);
});

view.addEventListener('mouseleave', () => {
  pointer = null;
  loupe.style.display = 'none';
  draw();
});

view.addEventListener('mouseup', e => {
  const d = dragging;
  dragging = null;
  if (!d) return;

  if (d.mode === 'new' && !d.moved && e.button === 0) {
    const r = stage.getBoundingClientRect();
    const ip = toImage(e.clientX - r.left, e.clientY - r.top);
    const p = el('snap').checked ? snapToStar(ip.x, ip.y) : ip;
    draftPoint = { x: p.x, y: p.y };
    selectedId = null;
    openEditor('');
    draw();
  } else if (d.mode === 'point' && !d.moved) {
    draftPoint = null;
    selectedId = d.star.id;
    openEditor(d.star.name);
    renderList();
    draw();
  }
});

view.addEventListener('contextmenu', e => e.preventDefault());

view.addEventListener('wheel', e => {
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const before = toImage(sx, sy);
  const factor = Math.exp(-e.deltaY * 0.0015);
  scale = Math.min(40, Math.max(0.05, scale * factor));
  offX = sx - before.x * scale;
  offY = sy - before.y * scale;
  pointer = toImage(sx, sy);
  draw();
  drawLoupe(sx, sy, pointer);
}, { passive: false });

window.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.activeElement !== nameInput) { spaceDown = true; e.preventDefault(); }
});
window.addEventListener('keyup', e => { if (e.code === 'Space') spaceDown = false; });

// --- editor --------------------------------------------------------------
function openEditor(name) {
  const p = draftPoint || stars.find(s => s.id === selectedId);
  if (!p) return;
  editor.classList.remove('hidden');
  el('editorCoord').textContent = `x ${p.x.toFixed(1)} · y ${p.y.toFixed(1)}`;
  el('deletePoint').style.display = draftPoint ? 'none' : '';
  nameInput.value = name || '';
  nameInput.focus();
  nameInput.select();
}

function closeEditor() {
  editor.classList.add('hidden');
  draftPoint = null;
  selectedId = null;
  renderList();
  draw();
}

function commitName() {
  const name = nameInput.value.trim();
  if (!name) { nameInput.focus(); return; }
  if (draftPoint) {
    stars.push({ id: nextId++, name, x: draftPoint.x, y: draftPoint.y });
  } else {
    const s = stars.find(s => s.id === selectedId);
    if (s) s.name = name;
  }
  closeEditor();
  saveLabels();
}

el('saveName').addEventListener('click', commitName);
el('cancelName').addEventListener('click', closeEditor);
el('deletePoint').addEventListener('click', () => {
  stars = stars.filter(s => s.id !== selectedId);
  closeEditor();
  saveLabels();
});

nameInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); commitName(); }
  if (e.key === 'Escape') { e.preventDefault(); closeEditor(); }
});

// --- list ----------------------------------------------------------------
function renderList() {
  el('count').textContent = stars.length;
  el('list').innerHTML = stars.map((s, i) => `
    <li data-id="${s.id}" class="${s.id === selectedId ? 'active' : ''}">
      <span class="idx">${i + 1}</span>
      <span class="nm">${escapeHtml(s.name)}</span>
      <span class="xy">${s.x.toFixed(1)}, ${s.y.toFixed(1)}</span>
    </li>`).join('');
}

el('list').addEventListener('click', e => {
  const li = e.target.closest('li');
  if (!li) return;
  const s = stars.find(s => s.id === Number(li.dataset.id));
  if (!s) return;
  draftPoint = null;
  selectedId = s.id;
  const r = stage.getBoundingClientRect();
  scale = Math.max(scale, 4);
  offX = r.width / 2 - s.x * scale;
  offY = r.height / 2 - s.y * scale;
  openEditor(s.name);
  renderList();
  draw();
});

const escapeHtml = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// --- persistence ---------------------------------------------------------
let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveLabels, 400);
}

async function saveLabels() {
  const payload = {
    image: IMAGE_NAME,
    width: img?.width ?? null,
    height: img?.height ?? null,
    updated: new Date().toISOString(),
    stars: stars.map(s => ({ name: s.name, x: +s.x.toFixed(2), y: +s.y.toFixed(2) }))
  };
  try {
    const res = await fetch('/api/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(res.statusText);
    flash(`已保存 ${stars.length} 个点`);
  } catch (err) {
    flash('保存失败: ' + err.message);
  }
  renderList();
}

async function loadLabels() {
  try {
    const res = await fetch('/api/labels');
    const data = await res.json();
    if (Array.isArray(data.stars)) {
      stars = data.stars.map(s => ({ id: nextId++, name: s.name, x: s.x, y: s.y }));
    }
  } catch { /* first run, no labels yet */ }
  renderList();
}

let flashTimer = null;
function flash(msg) {
  el('status').textContent = msg;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { el('status').textContent = ''; }, 2200);
}

// --- export --------------------------------------------------------------
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

el('exportCsv').addEventListener('click', () => {
  const rows = ['name,x,y', ...stars.map(s => `"${s.name.replace(/"/g, '""')}",${s.x.toFixed(2)},${s.y.toFixed(2)}`)];
  download('star_labels.csv', rows.join('\n'), 'text/csv');
});

el('exportJson').addEventListener('click', () => {
  download('star_labels.json', JSON.stringify({
    image: IMAGE_NAME,
    stars: stars.map(s => ({ name: s.name, x: +s.x.toFixed(2), y: +s.y.toFixed(2) }))
  }, null, 2), 'application/json');
});

el('clearAll').addEventListener('click', () => {
  if (!stars.length || !confirm('清空所有标注？')) return;
  stars = [];
  closeEditor();
  saveLabels();
});
