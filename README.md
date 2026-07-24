# Image Enhancer

Real-ESRGAN-powered image upscaler / enhancer — web app with two backends:

- **Server-side (default)**: Real-ESRGAN_x4 ONNX on Flask + ONNX Runtime (CPU-friendly)
- **Browser-side (fallback)**: SwinIR via onnxruntime-web (runs entirely on the user's CPU)

## Features

- 4× super-resolution for photos (real-world, not anime)
- Drag-and-drop upload, max 30MB
- Live progress bar via Server-Sent Events (per-tile updates)
- Tile-based inference with Hann-window blending (no seam artifacts)
- Automatic mode selection: server if image ≤800px wide, browser if larger

## Performance (CPU-only)

| Input size | Tiles | Expected time |
|---|---|---|
| 111×167  | 1   | ~10 sec |
| 400×300  | 12  | ~50 sec |
| 600×400  | 24  | ~2 min  |
| 1024×1024 | 64 | ~6 min |

Browser-side is variable depending on the user's CPU.

## Stack

- **Backend**: Flask 3.x, Pillow, ONNX Runtime 1.27
- **Frontend**: Vanilla JS, Server-Sent Events, no bundler
- **Models**: Real-ESRGAN (xinntao), SwinIR (JingyunLiang) — both pre-exported to ONNX
- **Service**: runs as a systemd unit on port 5078, exposed via nginx on 8098

## Setup

1. Install dependencies in your Python venv:

```bash
pip install pillow onnxruntime numpy flask
```

2. Download the model (64 MB):

```bash
mkdir -p models
curl -L -o models/RealESRGAN_x4.onnx \
  https://huggingface.co/bukuroo/Real-ESRGAN-x4-ONNX/resolve/main/RealESRGAN_x4.onnx
```

   The repo `[bukuroo/Real-ESRGAN-x4-ONNX](https://huggingface.co/bukuroo/Real-ESRGAN-x4-ONNX)` mirrors the original `Meeperomi/RealESRGAN_x4-onnx` model. Both work — bukuroo's is more reliably hosted.

3. Run:

```bash
python3 enhancer.py
# Open http://localhost:5078/
```

## Layout

```
image-enhancer/
├── enhancer.py             # Flask backend (tile processing, SSE progress)
├── templates/index.html    # Drop zone + sidebar + preview
├── static/
│   ├── css/style.css       # Dark theme
│   └── js/enhancer.js      # Frontend (drag&drop, SSE, browser fallback)
├── models/                 # ONNX models go here (gitignored)
├── cache/                  # Generated PNGs (gitignored)
└── .gitignore
```

## API

### `POST /api/enhance`

Start an enhancement job. Multipart form:

- `image`: image file (jpg/png/webp)
- `model`: model key (currently only `realesrgan-x4`)

Returns:

```json
{
  "job_id": "uuid",
  "width": 111,
  "height": 167,
  "total_tiles": 1,
  "model": "realesrgan-x4",
  "model_scale": 4
}
```

### `GET /api/progress/<job_id>`

Server-Sent Events stream. Emits a JSON payload each ~500ms:

```json
{
  "done": 5,
  "total": 12,
  "message": "Procesando tile 5/12...",
  "percent": 41.6,
  "finished": false,
  "result_url": null,
  "error": null
}
```

When `finished: true`, `result_url` holds the path to download the PNG.

### `GET /api/result/<name>`

Download a previously generated PNG.

### `GET /api/health`

Status of loaded models and cache size.

## Browser-side fallback

For very large images, switch to "Browser" mode in the UI. The page will:

1. Load `onnxruntime-web` from jsdelivr (CDN, ~1MB wasm)
2. Download the SwinIR ONNX model from Hugging Face (**58 MB**, first time only — cached by the browser)
3. Show a live progress bar during download (MB/% counter)
4. Run inference entirely on the user's CPU
5. Display the result without any server roundtrip

This keeps the VPS CPU/RAM free for other users, at the cost of a longer first load.

Performance (browser-side, varies by user CPU):

| Input size | Browser time (typical) |
|---|---|
| 64×64   | ~10 sec  |
| 96×96   | ~22 sec  |
| 256×256 | ~3-5 min |

Browser-side inference is ~5-10× slower than server-side per pixel, but it scales linearly with the image area. For images >800px wide, the freedom from server queue makes it worth it.

## Recent fixes

- **Fixed SwinIR URL** — was pointing to a non-existent file (`swin_ir_onnx.onnx`, 404). Now correctly downloads `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx` (58 MB). With live download progress.
- **Output clamping** — SwinIR can produce values slightly outside [0,1]; now clamp + round aggressively to prevent invalid pixels.
- **Download button reorder** — `setBusyButtons(false)` was disabling the download button after it was re-enabled; now download stays enabled after success.

## License

MIT (this codebase). The pre-trained models have their own licenses:
- Real-ESRGAN: BSD-3-Clause (xinntao)
- SwinIR: Apache-2.0 (JingyunLiang)

## Credits

- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — the upscaler
- [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) — alternative model
- [xinntao/Real-ESRGAN-ncnn-Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-Vulkan) — model zoo
- [Meeperomi/RealESRGAN_x4-onnx](https://huggingface.co/Meeperomi/RealESRGAN_x4-onnx) — pre-exported ONNX
