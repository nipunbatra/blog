// Sapiens2 try-it — vanilla JS frontend.

const TASK_BLURB = {
  pose: '308 keypoints (body + face + hands + feet). Hover any dot for its name + score.',
  seg:  '29-class body-part segmentation. Hover any colored region for its class.',
  normal:   'Surface normals as RGB (XYZ → R G B). Smooth = flat, banded = curvature.',
  pointmap: 'Per-pixel XYZ depth. Visualised as percentile-normalised turbo colormap.',
};

const state = {
  task: 'pose',
  vtask: 'pose',
  selectedSample: null,
  uploadedFile: null,
  selectedVideo: null,
  uploadedVideo: null,
  segLabels: null,        // Image element with labels-as-png
  segLabelsCanvas: null,  // offscreen canvas to read pixel values
  segClassNames: null,
  poseImageNatural: null, // {w, h}
};

// ---------- tabs ----------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b =>
      b.setAttribute('aria-selected', String(b === btn)));
    document.querySelectorAll('[data-panel]').forEach(p =>
      p.hidden = p.dataset.panel !== btn.dataset.tab);
  });
});

// ---------- task chips ----------
function setTask(taskKey, attr, blurbEl) {
  return () => {
    document.querySelectorAll(`[${attr}]`).forEach(b => {
      const sel = b.getAttribute(attr) === taskKey;
      b.setAttribute('aria-pressed', String(sel));
    });
    if (blurbEl) blurbEl.textContent = TASK_BLURB[taskKey];
  };
}
document.querySelectorAll('[data-task]').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.disabled) return;
    state.task = btn.dataset.task;
    setTask(state.task, 'data-task', document.getElementById('task-blurb'))();
  });
});
document.querySelectorAll('[data-vtask]').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.disabled) return;
    state.vtask = btn.dataset.vtask;
    setTask(state.vtask, 'data-vtask', null)();
  });
});
document.getElementById('task-blurb').textContent = TASK_BLURB.pose;

// ---------- video fps slider ----------
const fpsSlider = document.getElementById('video-fps');
const fpsVal    = document.getElementById('video-fps-val');
fpsSlider.addEventListener('input', () => fpsVal.textContent = (+fpsSlider.value).toFixed(1));

// ---------- load samples ----------
async function loadImageSamples() {
  const res = await fetch('/api/samples');
  const items = await res.json();
  const groups = {};
  items.forEach(it => (groups[it.group] = groups[it.group] || []).push(it));
  const wrap = document.getElementById('image-samples');
  wrap.innerHTML = '';
  for (const [name, lst] of Object.entries(groups)) {
    const sec = document.createElement('div');
    sec.innerHTML = `<div class="text-xs uppercase tracking-wider text-ink-500 mb-2">${name}</div>`;
    const grid = document.createElement('div');
    grid.className = 'grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2';
    lst.forEach(it => {
      const card = document.createElement('div');
      card.className = 'sample-card bg-white rounded shadow-sm overflow-hidden border border-ink-100';
      card.dataset.fname = it.fname;
      card.innerHTML = `
        <img src="${it.url}" alt="${it.desc}" class="w-full h-24 object-cover bg-ink-100">
        <div class="text-[11px] text-ink-700 px-2 py-1 truncate" title="${it.desc} · ${it.fname}">${it.desc}</div>`;
      card.addEventListener('click', () => selectImageSample(it.fname, it.url, card));
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    wrap.appendChild(sec);
  }
}

async function loadVideoSamples() {
  const res = await fetch('/api/videos');
  const items = await res.json();
  const grid = document.getElementById('video-samples');
  grid.innerHTML = '';
  items.forEach(it => {
    const card = document.createElement('div');
    card.className = 'sample-card bg-white rounded shadow-sm overflow-hidden border border-ink-100';
    card.dataset.fname = it.fname;
    card.innerHTML = `
      <img src="${it.thumb}" alt="${it.desc}" class="w-full h-32 object-cover bg-ink-900">
      <div class="text-[11px] text-ink-700 px-2 py-1.5 truncate" title="${it.desc}">${it.desc}</div>`;
    card.addEventListener('click', () => selectVideoSample(it.fname, it.url, card));
    grid.appendChild(card);
  });
}

function selectImageSample(fname, url, card) {
  state.selectedSample = fname;
  state.uploadedFile = null;
  document.querySelectorAll('[data-panel="image"] .sample-card').forEach(c =>
    c.classList.toggle('selected', c === card));
  const wrap = document.getElementById('image-input-wrap');
  wrap.innerHTML = `<img src="${url}" class="max-h-[520px] mx-auto rounded">`;
}

function selectVideoSample(fname, url, card) {
  state.selectedVideo = fname;
  state.uploadedVideo = null;
  document.querySelectorAll('[data-panel="video"] .sample-card').forEach(c =>
    c.classList.toggle('selected', c === card));
  const v = document.getElementById('video-input-el');
  v.src = url;
  document.getElementById('video-input-name').textContent = fname;
}

document.getElementById('image-upload').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  state.uploadedFile = f;
  state.selectedSample = null;
  document.querySelectorAll('[data-panel="image"] .sample-card').forEach(c =>
    c.classList.remove('selected'));
  const wrap = document.getElementById('image-input-wrap');
  const url = URL.createObjectURL(f);
  wrap.innerHTML = `<img src="${url}" class="max-h-[520px] mx-auto rounded">`;
});

document.getElementById('video-upload').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  state.uploadedVideo = f;
  state.selectedVideo = null;
  document.querySelectorAll('[data-panel="video"] .sample-card').forEach(c =>
    c.classList.remove('selected'));
  const v = document.getElementById('video-input-el');
  v.src = URL.createObjectURL(f);
  document.getElementById('video-input-name').textContent = f.name;
});

// ---------- error toast ----------
const errPanel  = document.getElementById('error-panel');
const errSum    = document.getElementById('error-summary');
const errTrace  = document.getElementById('error-trace');
document.getElementById('error-close').addEventListener('click', () => errPanel.hidden = true);
document.getElementById('error-copy').addEventListener('click', () => {
  navigator.clipboard.writeText(errSum.textContent + '\n\n' + errTrace.textContent);
});
function showError(payload) {
  errSum.innerHTML = `<strong>${payload.type || 'Error'}</strong> in <code>${payload.task || '—'}</code>: ${payload.error || payload.detail || 'unknown'}`;
  errTrace.textContent = payload.traceback || '(no traceback)';
  errPanel.hidden = false;
}
function clearError() { errPanel.hidden = true; }

// ---------- run image ----------
const imageStatus = document.getElementById('image-status');
function setBusy(el, label) {
  el.innerHTML = label
    ? `<div class="spinner"></div> <span>${label}</span>`
    : '';
}

document.getElementById('image-run').addEventListener('click', async () => {
  if (!state.selectedSample && !state.uploadedFile) {
    imageStatus.innerHTML = '<span class="text-brand-500">Pick a sample or upload an image first.</span>';
    return;
  }
  setBusy(imageStatus, `Running ${state.task} · loading model on first use takes ~15 s…`);

  const fd = new FormData();
  fd.append('task', state.task);
  if (state.uploadedFile) fd.append('file', state.uploadedFile);
  else                    fd.append('sample', state.selectedSample);

  clearError();
  let res, data;
  try {
    res = await fetch('/api/predict/image', { method: 'POST', body: fd });
    data = await res.json();
  } catch (e) {
    setBusy(imageStatus, '');
    showError({ type: 'NetworkError', error: String(e), task: state.task });
    imageStatus.innerHTML = `<span class="text-brand-500">Network error.</span>`;
    return;
  }
  setBusy(imageStatus, '');
  if (!res.ok || data.error) {
    showError(data);
    imageStatus.innerHTML = `<span class="text-brand-500">${data.type || 'Error'} (see toast)</span>`;
    refreshStatus();
    return;
  }
  imageStatus.innerHTML =
    `<span class="text-ink-500">forward ${data.forward_s.toFixed(2)}s · wall ${data.wall_s.toFixed(2)}s</span>`;
  renderImageResult(data);
  refreshStatus();
});

function renderImageResult(d) {
  const wrap = document.getElementById('image-output-wrap');
  wrap.innerHTML = '';
  const meta = document.getElementById('image-meta');
  meta.innerHTML = '';

  if (d.task === 'pose') {
    renderPose(wrap, d);
    meta.innerHTML = `<strong>${d.kpts_above_thr}/${d.total_kpts}</strong> keypoints above threshold. ` +
                     `<span class="text-ink-500">Hover any dot for its name + score.</span>`;
  } else if (d.task === 'seg') {
    renderSeg(wrap, d);
    const tops = d.top_classes.slice(0, 5).map(t =>
      `${t.name}<span class="text-ink-500">(${t.px.toLocaleString()}px)</span>`).join(' · ');
    meta.innerHTML = `Foreground <strong>${d.fg_pct.toFixed(1)}%</strong>. Top: ${tops}. ` +
                     `<span class="text-ink-500">Hover any region for its body-part class.</span>`;
  } else {
    const img = document.createElement('img');
    img.src = d.vis;
    img.className = 'mx-auto max-h-[520px] rounded';
    wrap.appendChild(img);
    if (d.meta) {
      meta.innerHTML = '<code class="text-xs text-ink-500">' +
        Object.entries(d.meta).map(([k,v]) =>
          `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`).join('  ') + '</code>';
    }
  }
}

// ---------- pose render ----------
function renderPose(wrap, d) {
  const inputImg = new Image();
  inputImg.onload = () => {
    const W = inputImg.naturalWidth, H = inputImg.naturalHeight;
    // Container preserves aspect ratio and fits in available width.
    const container = document.createElement('div');
    container.style.cssText = `position:relative; max-width:100%;`;
    container.style.aspectRatio = `${W} / ${H}`;
    const img = document.createElement('img');
    img.src = d.input;
    img.style.cssText = 'width:100%; height:100%; display:block; border-radius:4px;';
    container.appendChild(img);

    // SVG overlay for skeleton + keypoints (uses image's coordinate system).
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; pointer-events:auto;';

    // Score → color (green ≥ 0.7, amber ≥ 0.5, otherwise red-ish)
    const colorOf = s =>
      s >= 0.7 ? '#22c55e' : s >= 0.5 ? '#f59e0b' : '#ef4444';

    // Skeleton (only if both endpoints are present)
    const kpById = {};
    d.keypoints.forEach(k => kpById[k.id] = k);
    d.skeleton.forEach(([a, b]) => {
      if (!kpById[a] || !kpById[b]) return;
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', kpById[a].x); ln.setAttribute('y1', kpById[a].y);
      ln.setAttribute('x2', kpById[b].x); ln.setAttribute('y2', kpById[b].y);
      ln.setAttribute('stroke', 'rgba(255,255,255,.85)');
      ln.setAttribute('stroke-width', Math.max(2, W / 400));
      svg.appendChild(ln);
    });

    // Keypoints
    const tooltip = document.getElementById('pose-tooltip');
    d.keypoints.forEach(k => {
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', k.x); c.setAttribute('cy', k.y);
      c.setAttribute('r', k.is_body ? Math.max(5, W / 180) : Math.max(3, W / 320));
      c.setAttribute('fill', colorOf(k.score));
      c.setAttribute('stroke', 'rgba(0,0,0,.5)');
      c.setAttribute('stroke-width', 0.7);
      c.style.cursor = 'crosshair';
      c.addEventListener('mouseenter', e => {
        tooltip.hidden = false;
        tooltip.textContent = `${k.name}  ·  ${k.score.toFixed(2)}`;
      });
      c.addEventListener('mousemove', e => {
        tooltip.style.left = (e.pageX) + 'px';
        tooltip.style.top  = (e.pageY) + 'px';
      });
      c.addEventListener('mouseleave', () => tooltip.hidden = true);
      svg.appendChild(c);
    });
    container.appendChild(svg);
    wrap.appendChild(container);
  };
  inputImg.src = d.input;
}

// ---------- seg render ----------
function renderSeg(wrap, d) {
  const inputImg = new Image();
  inputImg.onload = () => {
    const W = inputImg.naturalWidth, H = inputImg.naturalHeight;
    const container = document.createElement('div');
    container.style.cssText = 'position:relative; max-width:100%;';
    container.style.aspectRatio = `${W} / ${H}`;
    const visImg = document.createElement('img');
    visImg.src = d.vis;
    visImg.style.cssText = 'width:100%; height:100%; display:block; border-radius:4px;';
    container.appendChild(visImg);

    // Off-screen labels canvas (1 px == 1 class id)
    const labelsImg = new Image();
    labelsImg.onload = () => {
      const cv = document.createElement('canvas');
      cv.width = W; cv.height = H;
      cv.getContext('2d').drawImage(labelsImg, 0, 0);
      state.segLabelsCanvas = cv;
      state.segClassNames = d.class_names;
    };
    labelsImg.src = d.labels_png;

    // Hover surface
    const hover = document.createElement('div');
    hover.style.cssText = 'position:absolute; inset:0; cursor:crosshair;';
    const tooltip = document.getElementById('seg-tooltip');
    hover.addEventListener('mousemove', e => {
      if (!state.segLabelsCanvas) return;
      const rect = container.getBoundingClientRect();
      const px = Math.floor((e.clientX - rect.left) / rect.width * W);
      const py = Math.floor((e.clientY - rect.top)  / rect.height * H);
      const data = state.segLabelsCanvas.getContext('2d')
                       .getImageData(px, py, 1, 1).data;
      const cls = data[0];
      const name = state.segClassNames[cls] || (cls === 0 ? 'background' : `class_${cls}`);
      tooltip.hidden = false;
      tooltip.textContent = name;
      tooltip.style.left = e.pageX + 'px';
      tooltip.style.top  = e.pageY + 'px';
    });
    hover.addEventListener('mouseleave', () => tooltip.hidden = true);
    container.appendChild(hover);
    wrap.appendChild(container);
  };
  inputImg.src = d.input;
}

// ---------- run video ----------
const videoStatus = document.getElementById('video-status');
document.getElementById('video-run').addEventListener('click', async () => {
  if (!state.selectedVideo && !state.uploadedVideo) {
    videoStatus.innerHTML = '<span class="text-brand-500">Pick a clip or upload one first.</span>';
    return;
  }
  const fps = +fpsSlider.value;
  setBusy(videoStatus, `Running ${state.vtask} on every ${(1/fps).toFixed(1)}s of the clip…`);
  document.getElementById('video-output-info').textContent = '';

  const fd = new FormData();
  fd.append('task', state.vtask);
  fd.append('fps', fps);
  if (state.uploadedVideo) fd.append('file', state.uploadedVideo);
  else                     fd.append('sample', state.selectedVideo);

  clearError();
  let res, data;
  try {
    res = await fetch('/api/predict/video', { method: 'POST', body: fd });
    data = await res.json();
  } catch (e) {
    setBusy(videoStatus, '');
    showError({ type: 'NetworkError', error: String(e), task: state.vtask });
    videoStatus.innerHTML = `<span class="text-brand-500">Network error.</span>`;
    return;
  }
  setBusy(videoStatus, '');
  if (!res.ok || data.error) {
    showError(data);
    videoStatus.innerHTML = `<span class="text-brand-500">${data.type || 'Error'} (see toast)</span>`;
    refreshStatus();
    return;
  }
  const vo = document.getElementById('video-output-el');
  vo.src = data.video_url;
  vo.load();
  document.getElementById('video-output-info').textContent =
    `${data.frames} frames at ${data.sample_fps} fps · processed in ${data.wall_s.toFixed(1)}s`;
  refreshStatus();
});

// ---------- device info ----------
async function refreshStatus() {
  try {
    const s = await (await fetch('/api/status')).json();
    const loaded = s.loaded_models.length
      ? s.loaded_models.join(', ') : '(none)';
    document.getElementById('device-info').innerHTML =
      `<span>GPU <strong>${s.gpu_free_gb}</strong>/${s.gpu_total_gb} GB free</span>
       · loaded: <span class="text-ink-700">${loaded}</span>
       · disabled: ${s.disabled_tasks.join(', ') || 'none'}`;
  } catch {}
}
refreshStatus();
setInterval(refreshStatus, 8000);

// ---------- boot ----------
loadImageSamples();
loadVideoSamples();
