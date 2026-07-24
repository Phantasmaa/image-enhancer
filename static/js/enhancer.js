// Image Enhancer frontend
// Drag & drop → upload → progress via Server-Sent Events → download
// Plus a browser-side fallback using onnxruntime-web + SwinIR
// Author: Hermes (MiniMax-M3) for Manuel

(function () {
  'use strict';

  // ─── State ───────────────────────────────────────────────────────────
  const state = {
    file: null,           // File object
    dataURL: null,        // for client-side upscaling
    previewURL: null,     // for the original <img>
    serverResultURL: null,
    serverJobID: null,
    eventSource: null,
    browserSession: null,
    browserModel: null,
    busy: false,
  };

  // ─── Init ────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', init);

  function init() {
    showDropZone();
    bindUI();
    bindDropZone();
    checkServerHealth();
  }

  // ─── UI helpers ──────────────────────────────────────────────────────
  function setStatus(msg, kind = 'muted') {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'status small ' + kind;
  }

  function showDropZone() {
    document.getElementById('dropZone').style.display = 'flex';
    document.getElementById('workArea').style.display = 'none';
  }

  function showWorkArea() {
    document.getElementById('dropZone').style.display = 'none';
    document.getElementById('workArea').style.display = 'block';
  }

  function setProgress(done, total, message, percent) {
    document.getElementById('progressArea').style.display = 'block';
    document.getElementById('progressMessage').textContent = message;
    document.getElementById('progressPercent').textContent = Math.round(percent) + '%';
    document.getElementById('progressFill').style.width = Math.min(100, percent) + '%';
    if (done >= total && total > 0) {
      setTimeout(() => {
        document.getElementById('progressArea').style.display = 'none';
      }, 1200);
    }
  }

  function hideProgress() {
    document.getElementById('progressArea').style.display = 'none';
  }

  // ─── Server health check ─────────────────────────────────────────────
  async function checkServerHealth() {
    try {
      const r = await fetch('/api/health');
      const j = await r.json();
      const models = Object.entries(j.models)
        .map(([k, v]) => `${k}: ${v ? '✓' : '✗'}`)
        .join(' · ');
      document.getElementById('serverStatus').textContent =
        `OK · modelos: ${models}`;
    } catch (e) {
      document.getElementById('serverStatus').textContent =
        'ERROR · el server no responde';
    }
  }

  // ─── Bindings ────────────────────────────────────────────────────────
  function bindUI() {
    document.getElementById('enhanceBtn').addEventListener('click', startEnhance);
    document.getElementById('downloadBtn').addEventListener('click', downloadResult);
    document.getElementById('cancelBtn').addEventListener('click', cancelJob);
    document.getElementById('clearBtn').addEventListener('click', clearImage);
  }

  function bindDropZone() {
    const dz = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInputDrop');
    dz.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (f) loadFile(f);
    });
    ['dragenter', 'dragover'].forEach(evt =>
      dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add('drag-over'); }));
    ['dragleave', 'drop'].forEach(evt =>
      dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove('drag-over'); }));
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith('image/')) loadFile(f);
    });
  }

  // ─── Load image ──────────────────────────────────────────────────────
  async function loadFile(file) {
    if (state.busy) return;
    if (file.size > 30 * 1024 * 1024) {
      setStatus('Imagen demasiado grande (máx 30 MB)', 'err');
      return;
    }
    state.file = file;
    state.dataURL = await fileToDataURL(file);
    state.previewURL = state.dataURL;
    state.serverResultURL = null;
    state.serverJobID = null;
    state.browserSession = null;
    state.browserModel = null;

    document.getElementById('originalImg').src = state.previewURL;
    document.getElementById('enhancedImg').style.display = 'none';
    document.getElementById('enhancedPlaceholder').style.display = 'block';

    // Show metadata
    const img = new Image();
    img.onload = () => {
      const model = document.getElementById('modelSelect');
      const w = img.naturalWidth, h = img.naturalHeight;
      const tileCount = Math.ceil(w / 168) * Math.ceil(h / 168);
      const secEst = tileCount * 6;
      document.getElementById('imgInfo').textContent =
        `${file.name} · ${w}×${h} · ${(file.size/1024).toFixed(0)} KB · ~${Math.ceil(secEst/60)} min estimado (server)`;
      document.getElementById('clearBtn').disabled = false;
      document.getElementById('enhanceBtn').disabled = false;
      document.getElementById('downloadBtn').disabled = true;
    };
    img.src = state.previewURL;
    showWorkArea();
    setStatus('Imagen cargada · elegí modo y hace click en "Mejorar"', 'ok');
  }

  function clearImage() {
    if (state.busy) return;
    if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
    state.file = null;
    state.dataURL = null;
    state.previewURL = null;
    state.serverResultURL = null;
    state.serverJobID = null;
    document.getElementById('imgInfo').textContent = 'Sin imagen';
    document.getElementById('clearBtn').disabled = true;
    document.getElementById('enhanceBtn').disabled = true;
    document.getElementById('downloadBtn').disabled = true;
    document.getElementById('enhancedImg').style.display = 'none';
    document.getElementById('enhancedPlaceholder').style.display = 'block';
    hideProgress();
    setStatus('Imagen quitada', 'muted');
    showDropZone();
  }

  function fileToDataURL(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsDataURL(file);
    });
  }

  // ─── Decide mode ─────────────────────────────────────────────────────
  function pickMode() {
    const sel = document.getElementById('modeSelect').value;
    if (sel === 'server') return 'server';
    if (sel === 'browser') return 'browser';
    // auto: decide by image size
    const img = document.getElementById('originalImg');
    const w = img.naturalWidth || 99999;
    return w > 800 ? 'browser' : 'server';
  }

  // ─── Start enhance ───────────────────────────────────────────────────
  async function startEnhance() {
    if (!state.file || state.busy) return;
    state.busy = true;
    setBusyButtons(true);
    hideProgress();

    const mode = pickMode();
    if (mode === 'server') {
      await enhanceServer();
    } else {
      await enhanceBrowser();
    }
  }

  function setBusyButtons(busy) {
    document.getElementById('enhanceBtn').disabled = busy;
    document.getElementById('cancelBtn').disabled = !busy;
    document.getElementById('downloadBtn').disabled = true;
    document.getElementById('clearBtn').disabled = busy;
  }

  // ─── Server-side enhance ─────────────────────────────────────────────
  async function enhanceServer() {
    setStatus('Subiendo al servidor...', 'warn');
    setProgress(0, 100, 'Subiendo...', 0);

    const model = document.getElementById('modelSelect').value;
    const fd = new FormData();
    fd.append('image', state.file);
    fd.append('model', model);

    let resp;
    try {
      resp = await fetch('/api/enhance', { method: 'POST', body: fd });
    } catch (e) {
      cleanup('Error de red: ' + e.message, 'err');
      return;
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      cleanup('Error: ' + err.error, 'err');
      return;
    }
    const job = await resp.json();
    state.serverJobID = job.job_id;
    setProgress(0, job.total_tiles,
                `Iniciando realce · ${job.width}×${job.height} · ${job.total_tiles} tiles...`,
                0);
    setStatus(`Procesando ${job.total_tiles} tiles en el servidor...`, 'warn');

    // Subscribe to progress via SSE
    state.eventSource = new EventSource(`/api/progress/${job.job_id}`);
    state.eventSource.onmessage = (e) => {
      const p = JSON.parse(e.data);
      const pct = p.percent || 0;
      setProgress(p.done, p.total, p.message, pct);
      if (p.finished) {
        state.eventSource.close();
        state.eventSource = null;
        if (p.error) {
          cleanup('Error en server: ' + p.error, 'err');
          return;
        }
        state.serverResultURL = p.result_url;
        document.getElementById('enhancedImg').src = p.result_url;
        document.getElementById('enhancedImg').style.display = 'block';
        document.getElementById('enhancedPlaceholder').style.display = 'none';
        setStatus(`Listo · ${p.message}`, 'ok');
        state.busy = false;
        // Re-enable buttons AFTER state.busy = false so setBusyButtons(false)
        // doesn't immediately disable the download button again
        setBusyButtons(false);
        document.getElementById('downloadBtn').disabled = false;
      }
    };
    state.eventSource.onerror = () => {
      if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
        if (state.busy) {
          cleanup('Conexión de progreso perdida', 'err');
        }
      }
    };
  }

  function cancelJob() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    state.busy = false;
    setBusyButtons(false);
    hideProgress();
    setStatus('Cancelado', 'warn');
  }

  function cleanup(msg, kind) {
    state.busy = false;
    setBusyButtons(false);
    hideProgress();
    setStatus(msg, kind);
  }

  // ─── Browser-side fallback ───────────────────────────────────────────
  // Uses onnxruntime-web + a SwinIR-style super-resolution model.
  // Loads from huggingface.co at first use.
  //
  // Model: rocca/swin-ir-onnx → 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx
  //   4× super-resolution model trained with BSRGAN degradation on real photos.
  //   Input: float32 [1, 3, H, W] in range [0, 1] · output: same shape ×4.
  const SWINIR_MODEL_URL =
    'https://huggingface.co/rocca/swin-ir-onnx/resolve/main/' +
    '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx';

  async function enhanceBrowser() {
    setStatus('Cargando motor de IA en el browser...', 'warn');
    setProgress(2, 100, 'Cargando onnxruntime-web...', 2);

    try {
      // Load onnxruntime-web from CDN
      if (!window.ort) {
        await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');
      }
      if (!window.ort) throw new Error('No se pudo cargar onnxruntime-web');

      // Configure WASM paths (required for multi-thread / SIMD fallback)
      window.ort.env.wasm.wasmPaths =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

      if (!state.browserSession) {
        setProgress(5, 100, 'Descargando modelo SwinIR (~58 MB, primera vez tarda)...', 5);
        state.browserSession = await window.ort.InferenceSession.create(
          SWINIR_MODEL_URL,
          {
            executionProviders: ['wasm'],
            progressCallback: (p) => {
              if (typeof p.loadedBytes === 'number' && typeof p.totalBytes === 'number') {
                const pct = 5 + (p.loadedBytes / p.totalBytes) * 35;
                const mb = (p.loadedBytes / 1024 / 1024).toFixed(1);
                const total = (p.totalBytes / 1024 / 1024).toFixed(1);
                setProgress(pct, 100,
                  `Descargando modelo: ${mb}/${total} MB (${pct.toFixed(0)}%)`, pct);
              }
            },
          }
        );
      }

      setProgress(42, 100, 'Modelo cargado · procesando imagen...', 42);

      // Load image to canvas
      const img = document.getElementById('originalImg');
      const W = img.naturalWidth;
      const H = img.naturalHeight;
      // For large images, resize to a reasonable max for browser-side
      const maxDim = 512;
      let procW = W, procH = H;
      if (W > maxDim || H > maxDim) {
        const s = Math.min(maxDim / W, maxDim / H);
        procW = Math.round(W * s);
        procH = Math.round(H * s);
      }
      // Ensure dims are multiples of model's stride (typically 8)
      procW = Math.max(8, Math.round(procW / 8) * 8);
      procH = Math.max(8, Math.round(procH / 8) * 8);

      const canvas = document.createElement('canvas');
      canvas.width = procW; canvas.height = procH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, procW, procH);

      const imgData = ctx.getImageData(0, 0, procW, procH);
      // CHW float32 [0,1] in NCHW [1, 3, H, W]
      const arr = new Float32Array(procW * procH * 3);
      for (let i = 0, j = 0; i < imgData.data.length; i += 4, j += 3) {
        arr[j    ] = imgData.data[i    ] / 255.0;
        arr[j + 1] = imgData.data[i + 1] / 255.0;
        arr[j + 2] = imgData.data[i + 2] / 255.0;
      }
      const tensor = new window.ort.Tensor('float32', arr, [1, 3, procH, procW]);

      setStatus(
        `Inferencia SwinIR ${procW}×${procH} → ${procW*4}×${procH*4} (puede tardar 1-3 min en CPU del browser)`,
        'warn'
      );
      setProgress(50, 100, 'Inferencia en CPU del browser...', 50);
      const t0 = performance.now();
      const out = await state.browserSession.run({ input: tensor });
      const elapsedSec = ((performance.now() - t0) / 1000).toFixed(1);
      console.log(`SwinIR inference: ${elapsedSec}s`);
      const outArr = out.output.data;
      const outDims = out.output.dims;
      const outH = outDims[outDims.length - 2];
      const outW = outDims[outDims.length - 1];

      setProgress(85, 100, 'Decodificando resultado...', 85);

      // CHW float → HWC uint8 with clamp (SwinIR outputs can drift outside [0,1])
      const outCanvas = document.createElement('canvas');
      outCanvas.width = outW;
      outCanvas.height = outH;
      const outCtx = outCanvas.getContext('2d');
      const outImgData = outCtx.createImageData(outW, outH);
      for (let i = 0, j = 0; j < outArr.length; i += 4, j += 3) {
        // The model can output values slightly outside [0,1]; clamp aggressively.
        outImgData.data[i    ] = Math.max(0, Math.min(255, Math.round(outArr[j    ] * 255)));
        outImgData.data[i + 1] = Math.max(0, Math.min(255, Math.round(outArr[j + 1] * 255)));
        outImgData.data[i + 2] = Math.max(0, Math.min(255, Math.round(outArr[j + 2] * 255)));
        outImgData.data[i + 3] = 255;
      }
      outCtx.putImageData(outImgData, 0, 0);

      const dataURL = outCanvas.toDataURL('image/png');
      document.getElementById('enhancedImg').src = dataURL;
      document.getElementById('enhancedImg').style.display = 'block';
      document.getElementById('enhancedPlaceholder').style.display = 'none';
      state.serverResultURL = dataURL;
      state.serverJobID = 'browser';
      document.getElementById('downloadBtn').disabled = false;
      setStatus(
        `Listo (browser) · ${procW}×${procH} → ${outW}×${outH} en ${elapsedSec}s`,
        'ok'
      );
      setProgress(100, 100, 'Completado', 100);
    } catch (e) {
      console.error('Browser enhance error:', e);
      cleanup('Error en browser: ' + (e.message || e), 'err');
      return;
    }
    state.busy = false;
    setBusyButtons(false);
    // setBusyButtons(false) disables the download button; re-enable now that
    // the operation truly completed (this is the last statement in the function).
    document.getElementById('downloadBtn').disabled = false;
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  // ─── Download result ─────────────────────────────────────────────────
  function downloadResult() {
    if (!state.serverResultURL) return;
    let url = state.serverResultURL;
    let name = 'enhanced.png';
    if (url.startsWith('/api/result/')) {
      url = window.location.origin + url;
    }
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // ─── Public API ──────────────────────────────────────────────────────
  window.imageEnhancer = { state, loadFile };
})();
