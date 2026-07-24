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
  https://huggingface.co/Meeperomi/RealESRGAN_x4-onnx/resolve/main/RealESRGAN_x4.onnx
```

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
2. Download the SwinIR ONNX model from Hugging Face (~12MB, cached after)
3. Run inference entirely on the user's CPU
4. Display the result without any server roundtrip

This keeps the VPS CPU/RAM free for other users, at the cost of a longer first load.

## License

MIT (this codebase). The pre-trained models have their own licenses:
- Real-ESRGAN: BSD-3-Clause (xinntao)
- SwinIR: Apache-2.0 (JingyunLiang)

## Credits

- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — the upscaler
- [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) — alternative model
- [xinntao/Real-ESRGAN-ncnn-Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-Vulkan) — model zoo
- [Meeperomi/RealESRGAN_x4-onnx](https://huggingface.co/Meeperomi/RealESRGAN_x4-onnx) — pre-exported ONNX
