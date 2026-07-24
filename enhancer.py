"""
Image Enhancer — Flask backend with Real-ESRGAN ONNX
Runs on :5078, exposed via nginx on :8098
Author: Hermes (MiniMax-M3) for Manuel
"""
import io
import os
import time
import uuid
import json
import base64
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

import numpy as np
from PIL import Image
import onnxruntime as ort

# ─── Paths ─────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / 'models'
CACHE_DIR = APP_ROOT / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

# ─── Model registry ────────────────────────────────────────────────────
# Each entry: {path, scale, type, description}
MODELS = {
    'realesrgan-x4': {
        'path': MODELS_DIR / 'RealESRGAN_x4.onnx',
        'scale': 4,
        'type': 'photo',
        'desc': 'Real-ESRGAN x4 (fotos reales, 64MB, ~6s por 200×200 tile en CPU)',
    },
}

# ─── Inference sessions (lazy load) ────────────────────────────────────
_sessions: dict = {}
_session_lock = threading.Lock()


def get_session(model_key: str):
    """Load ONNX inference session (cached)."""
    if model_key in _sessions:
        return _sessions[model_key]
    with _session_lock:
        if model_key in _sessions:
            return _sessions[model_key]
        info = MODELS[model_key]
        if not info['path'].exists():
            raise FileNotFoundError(f'Model file missing: {info["path"]}')
        sess = ort.InferenceSession(
            str(info['path']),
            providers=['CPUExecutionProvider'],
        )
        _sessions[model_key] = (sess, info)
        return _sessions[model_key]


# ─── Tile processing with overlap blending ──────────────────────────────
# Real-ESRGAN works best with inputs divisible by the network's stride.
# We tile large images into overlapping chunks and blend the overlaps
# with a raised-cosine (Hann) window to avoid seams.

HANN_CACHE = {}


def hann_window(tile_size: int, overlap: int) -> np.ndarray:
    """2D Hann window for tile blending. Returns (1,1,H,W) float32."""
    key = (tile_size, overlap)
    if key in HANN_CACHE:
        return HANN_CACHE[key]
    h = tile_size
    w = tile_size
    # 1D Hann across each dimension
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    if overlap > 0:
        t = np.linspace(0, np.pi, overlap, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(t)  # 0 → 1
        wy[:overlap] = ramp
        wy[-overlap:] = ramp[::-1]
        wx[:overlap] = ramp
        wx[-overlap:] = ramp[::-1]
    win = (wy[:, None] * wx[None, :]).astype(np.float32)
    win = win[None, None, :, :]
    HANN_CACHE[key] = win
    return win


def preprocess(pil_img: Image.Image) -> np.ndarray:
    """HWC uint8 RGB → 1×3×H×W float32 in [0,1]."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def postprocess(out: np.ndarray) -> Image.Image:
    """1×3×H×W float → PIL RGB uint8."""
    arr = out[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def infer_tile(sess, tile: np.ndarray) -> np.ndarray:
    """Run ONNX inference on a single tile (1×3×H×W float32)."""
    return sess.run(None, {'input': tile})[0]


def upscale_image(
    pil_img: Image.Image,
    sess,
    scale: int,
    progress_cb=None,
    cancel_event: threading.Event = None,
    tile_size: int = 128,
    overlap: int = 16,
) -> Image.Image:
    """Upscale image by `scale`× using tile processing with overlap blending.

    Args:
        pil_img: input PIL image (RGB)
        sess: ONNX inference session
        scale: 2 or 4
        progress_cb: fn(done_tiles, total_tiles, current_y) — called per tile
        cancel_event: if set, raises InterruptedError
        tile_size: input tile size in pixels (must be divisible by model's stride)
        overlap: pixels of overlap between adjacent tiles for blending
    """
    W, H = pil_img.size
    inp = preprocess(pil_img)

    # Stride = tile_size - overlap
    stride = tile_size - overlap

    # Compute tile grid
    nx = max(1, (W + stride - 1) // stride)
    ny = max(1, (H + stride - 1) // stride)
    total = nx * ny

    # Output canvas at 4× scale (or 2× for scale=2)
    out_canvas = np.zeros(
        (3, H * scale, W * scale),
        dtype=np.float32,
    )
    weight = np.zeros((H * scale, W * scale), dtype=np.float32)

    win = hann_window(tile_size * scale, overlap * scale)

    done = 0
    for ty in range(ny):
        y0 = ty * stride
        y1 = min(y0 + tile_size, H)
        # Pad last tile if needed (model needs fixed tile size)
        pad_bottom = tile_size - (y1 - y0)
        for tx in range(nx):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError('cancelled')
            x0 = tx * stride
            x1 = min(x0 + tile_size, W)
            pad_right = tile_size - (x1 - x0)

            # Extract tile (with padding if at edge)
            tile = inp[:, :, y0:y1, x0:x1]
            if pad_bottom or pad_right:
                tile = np.pad(
                    tile,
                    ((0, 0), (0, 0), (0, pad_bottom), (0, pad_right)),
                    mode='reflect',
                )

            # Run model on tile
            out_tile = infer_tile(sess, tile)

            # Crop back to actual tile output (drop padding)
            oh = (y1 - y0) * scale
            ow = (x1 - x0) * scale
            out_tile = out_tile[:, :, :oh, :ow]

            # Apply Hann window weight
            w = win[:, :, :oh, :ow]

            # Place into output canvas
            oy0 = y0 * scale
            ox0 = x0 * scale
            out_canvas[:, oy0:oy0 + oh, ox0:ox0 + ow] += (
                out_tile[0] * w[0, 0]
            )
            weight[oy0:oy0 + oh, ox0:ox0 + ow] += w[0, 0]

            done += 1
            if progress_cb:
                progress_cb(done, total, y0)

    # Normalize by accumulated weight
    weight = np.maximum(weight, 1e-8)
    out_canvas = out_canvas / weight[None, :, :]
    out_canvas = np.transpose(out_canvas, (1, 2, 0))
    out_canvas = np.clip(out_canvas * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out_canvas)


# ─── Progress tracking via SSE ─────────────────────────────────────────
# Each enhance job has a uuid → progress dict (in-memory)

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _set_progress(job_id: str, done: int, total: int, message: str):
    with _jobs_lock:
        _jobs[job_id] = {
            'done': done,
            'total': total,
            'message': message,
            'finished': False,
            'error': None,
            'result_url': None,
        }


def _finish_progress(job_id: str, result_url: str = None, error: str = None):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['finished'] = True
            _jobs[job_id]['result_url'] = result_url
            _jobs[job_id]['error'] = error


# ─── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB upload limit


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/static/<path:p>')
def static_files(p):
    return send_from_directory('static', p)


@app.route('/api/models')
def api_models():
    return jsonify({
        k: {'scale': v['scale'], 'type': v['type'], 'desc': v['desc'],
            'available': v['path'].exists()}
        for k, v in MODELS.items()
    })


@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    """Start an enhancement job. Returns {job_id}. Progress via /api/progress/<id>."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    model_key = request.form.get('model', 'realesrgan-x4')
    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400

    try:
        pil_img = Image.open(file.stream)
        pil_img.load()
    except Exception as e:
        return jsonify({'error': f'Could not read image: {e}'}), 400

    W, H = pil_img.size
    job_id = uuid.uuid4().hex

    # Run in background thread
    def run_job():
        try:
            sess, info = get_session(model_key)
            _set_progress(job_id, 0, 1, f'Cargando modelo {model_key}...')
            scale = info['scale']

            def cb(d, t, y):
                msg = f'Procesando tile {d}/{t} ({y}px / {H}px)...'
                _set_progress(job_id, d, t, msg)

            _set_progress(job_id, 0, 1, 'Iniciando realce...')
            t0 = time.time()
            result = upscale_image(pil_img, sess, scale, progress_cb=cb)
            elapsed = time.time() - t0

            # Save result to cache
            result_name = f'{job_id}.png'
            result_path = CACHE_DIR / result_name
            result.save(result_path, optimize=True)

            _finish_progress(
                job_id,
                result_url=f'/api/result/{result_name}',
                error=None,
            )
            # Final state for SSE clients
            _set_progress(job_id, 1, 1,
                          f'Listo en {elapsed:.1f}s · {W}×{H} → {result.size[0]}×{result.size[1]}')
            _finish_progress(job_id, result_url=f'/api/result/{result_name}')
        except InterruptedError:
            _finish_progress(job_id, error='Cancelado por el usuario')
        except Exception as e:
            _finish_progress(job_id, error=str(e))
            raise

    # Estimate tile count up-front for first progress message
    sess_info = MODELS[model_key]
    scale = sess_info['scale']
    tile_size = 128
    overlap = 16
    stride = tile_size - overlap
    nx = max(1, (W + stride - 1) // stride)
    ny = max(1, (H + stride - 1) // stride)
    total_tiles = nx * ny
    _set_progress(job_id, 0, total_tiles,
                  f'Imagen {W}×{H} · {total_tiles} tiles · cola...')

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return jsonify({
        'job_id': job_id,
        'width': W,
        'height': H,
        'total_tiles': total_tiles,
        'model': model_key,
        'model_scale': scale,
    })


@app.route('/api/progress/<job_id>')
def api_progress(job_id: str):
    """Server-Sent Events stream of job progress."""
    def event_stream():
        last_done = -1
        last_message = ''
        # Stream until finished
        for _ in range(60 * 30):  # max 30 min
            with _jobs_lock:
                p = _jobs.get(job_id)
            if p is None:
                yield f"event: error\ndata: {json.dumps({'error': 'unknown job'})}\n\n"
                return
            if p['done'] != last_done or p['message'] != last_message or p['finished']:
                last_done = p['done']
                last_message = p['message']
                payload = json.dumps({
                    'done': p['done'],
                    'total': p['total'],
                    'message': p['message'],
                    'percent': (p['done'] / p['total'] * 100) if p['total'] else 0,
                    'finished': p['finished'],
                    'result_url': p['result_url'],
                    'error': p['error'],
                })
                yield f"data: {payload}\n\n"
            if p['finished']:
                return
            time.sleep(0.5)
        yield "event: timeout\ndata: {}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/result/<name>')
def api_result(name):
    return send_from_directory(str(CACHE_DIR), name)


@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'ok',
        'models': {k: v['path'].exists() for k, v in MODELS.items()},
        'cached_sessions': list(_sessions.keys()),
        'cache_size': len(list(CACHE_DIR.glob('*.png'))),
    })


if __name__ == '__main__':
    # Pre-load the model so first request is fast
    print('Pre-loading models...')
    for k in MODELS:
        try:
            get_session(k)
            print(f'  {k}: ready')
        except Exception as e:
            print(f'  {k}: FAILED ({e})')
    print('Starting Flask on 0.0.0.0:5078...')
    app.run(host='0.0.0.0', port=5078, debug=False, threaded=True)
