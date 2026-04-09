# MNIST Dual Model Demo

A small web demo comparing **two approaches to handwritten digit recognition on MNIST**: **least-squares** multi-class classification (linear algebra) and a **ResNet** classifier (PyTorch). Users draw a digit on a canvas or upload an image; the API returns predictions and top-3 scores for each model.

**Live demo (Hugging Face Spaces):** [MNIST_Demo](https://huggingface.co/spaces/MinhTien30122003/MNIST_Demo) — opening the Space loads the interactive UI (`/` redirects to `/demo`).

---

## Features

- MNIST-style drawing canvas (black background, white strokes); default path for the demo.
- Optional PNG/JPEG upload.
- Side-by-side results for **Least Squares** and **ResNet** (prediction + top-3).
- REST API with Swagger docs at `/docs`.
- Custom color palette; CSS split into `theme` + `components` under `static/css/`.

---

## Architecture (summary)

| Component | Description |
|-----------|-------------|
| **Least Squares** | Weight matrix `W` of shape `(785, 10)` (784 pixels + bias term), saved as `models/least_squares_W.npy`. |
| **ResNet** | ResNet-style architecture (Bottleneck blocks), PyTorch `state_dict`, saved as `models/resnet_mnist_state_dict.pt`. |
| **Backend** | FastAPI + Uvicorn; images are preprocessed to grayscale 28×28 and normalized to `[0, 1]` in `inference.py`. |

---

## Project layout

```
Mnist_Project/
├── app.py              # FastAPI: routes, /demo page, static files
├── inference.py        # Load models + predict for both
├── requirements.txt
├── Dockerfile          # Docker deploy (HF Spaces: port 7860)
├── DEPLOY.md           # Extra deploy notes
├── models/
│   ├── least_squares_W.npy
│   └── resnet_mnist_state_dict.pt
└── static/css/
    ├── theme.css       # Color tokens + base typography
    └── components.css  # Cards, buttons, canvas, demo layout
```

---

## Local setup

**Requirements:** Python 3.11+ (recommended), `pip`.

```bash
cd Mnist_Project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place both model files under `models/` as above (or set environment variables below).

### Run the server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Browser: [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo) (or `/`, which redirects to `/demo`).
- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `RESNET_PATH` | Absolute path to `resnet_mnist_state_dict.pt` if not using the default `models/` location. |
| `LS_PATH` | Absolute path to `least_squares_W.npy` if not using the default `models/` location. |

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirects to `/demo`. |
| `GET` | `/demo` | Web demo (canvas + upload). |
| `GET` | `/api` | JSON: model load status + endpoint hints. |
| `GET` | `/health` | Health check. |
| `POST` | `/predict` | Single model: `file`, `model_name` (`resnet` \| `least_squares`). |
| `POST` | `/predict-both` | Both models: `file`. |

See `/docs` for request/response schemas.

---

## Deployment

- **Hugging Face Spaces (Docker):** see `DEPLOY.md` and `Dockerfile` (default port `7860`).
- Large model files: consider [Git LFS](https://git-lfs.com/) for `*.pt` / `*.npy`.

---

## Notes

- Canvas strokes are sent as PNG; the backend resizes to 28×28 to match the MNIST-style pipeline.
- If a model weight file is missing, the corresponding JSON field reports an explicit error.

---

## License

Educational / portfolio demo; source and weights are governed by your own repository terms.
