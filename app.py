from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import MnistInferenceService


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_DIR = BASE_DIR / "model"

# You can override these via environment variables.
default_resnet = MODELS_DIR / "resnet_mnist_state_dict.pt"
if not default_resnet.exists():
    default_resnet = MODEL_DIR / "resnet_mnist_state_dict.pt"

default_ls = MODELS_DIR / "least_squares_W.npy"
if not default_ls.exists():
    default_ls = MODEL_DIR / "least_squares_W.npy"

RESNET_PATH = Path(os.getenv("RESNET_PATH", str(default_resnet)))
LS_PATH = Path(os.getenv("LS_PATH", str(default_ls)))

service = MnistInferenceService(
    least_squares_weight_path=LS_PATH if LS_PATH.exists() else None,
    resnet_state_dict_path=RESNET_PATH if RESNET_PATH.exists() else None,
    device=None,
)

app = FastAPI(
    title="MNIST Classifier API",
    description="Predict handwritten digits using Least Squares or ResNet.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def root() -> dict:
    return {
        "message": "MNIST inference API is running.",
        "loaded_models": {
            "least_squares": service.least_squares is not None,
            "resnet": service.resnet is not None,
        },
        "usage": {
            "endpoint": "POST /predict",
            "fields": {
                "file": "image file (png/jpg/jpeg)",
                "model_name": "resnet | least_squares",
                "invert": "true | false",
            },
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MNIST Dual Model Demo</title>
  <link rel="stylesheet" href="/static/css/theme.css" />
  <link rel="stylesheet" href="/static/css/components.css" />
</head>
<body>
  <h1>MNIST Classifier Demo</h1>
  <div class="muted">Draw a digit in the square (default), then compare predictions from Least Squares and ResNet.</div>

  <div class="card">
    <div class="row">
      <div>
        <div style="margin-bottom: 8px; font-weight: 600;">Write your digit here</div>
        <canvas id="drawCanvas" width="280" height="280"></canvas>
        <div class="controls">
          <button id="predictCanvasBtn" type="button">Predict from drawing</button>
          <button id="clearCanvasBtn" class="secondary" type="button">Clear</button>
        </div>
      </div>
      <div id="previewWrap" style="display:none;">
        <div style="margin-bottom: 8px; font-weight: 600;">Preview sent to API</div>
        <img id="preview" alt="preview" style="display:block; background:#000;" />
      </div>
    </div>

    <details>
      <summary><strong>Optional:</strong> upload image instead</summary>
      <form id="uploadForm" style="margin-top:12px;">
        <div class="row">
          <input id="fileInput" type="file" name="file" accept="image/png,image/jpeg,image/jpg" />
          <button id="submitUploadBtn" type="submit">Predict from upload</button>
        </div>
      </form>
    </details>
  </div>

  <div id="status" class="card muted">Waiting for input...</div>
  <div id="results" class="result-grid"></div>

  <script>
    const canvas = document.getElementById("drawCanvas");
    const ctx = canvas.getContext("2d");
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const statusBox = document.getElementById("status");
    const resultsBox = document.getElementById("results");
    const preview = document.getElementById("preview");
    const previewWrap = document.getElementById("previewWrap");
    const predictCanvasBtn = document.getElementById("predictCanvasBtn");
    const clearCanvasBtn = document.getElementById("clearCanvasBtn");
    const submitUploadBtn = document.getElementById("submitUploadBtn");

    function resetCanvas() {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    resetCanvas();

    function setDrawStyle() {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 18;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
    }
    setDrawStyle();

    let drawing = false;
    function getPos(e) {
      const rect = canvas.getBoundingClientRect();
      const touch = e.touches && e.touches[0];
      const clientX = touch ? touch.clientX : e.clientX;
      const clientY = touch ? touch.clientY : e.clientY;
      return {
        x: (clientX - rect.left) * (canvas.width / rect.width),
        y: (clientY - rect.top) * (canvas.height / rect.height),
      };
    }

    function startDraw(e) {
      drawing = true;
      const p = getPos(e);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      e.preventDefault();
    }

    function draw(e) {
      if (!drawing) return;
      const p = getPos(e);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      e.preventDefault();
    }

    function stopDraw() {
      drawing = false;
      ctx.closePath();
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    window.addEventListener("touchend", stopDraw, { passive: false });

    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
      previewWrap.style.display = "block";
    });

    function renderModelCard(title, data) {
      if (data.error) {
        return `<div class="card"><h3>${title}</h3><div class="err">${data.error}</div></div>`;
      }
      const top3 = (data.top3 || []).map(
        (x, idx) => `${idx + 1}. ${x.digit}: ${Number(x.score).toFixed(4)}`
      ).join("<br/>");
      return `
        <div class="card">
          <h3>${title}</h3>
          <div class="pred ok">${data.predicted_digit}</div>
          <div class="top3">${top3}</div>
        </div>
      `;
    }

    async function runPredictWithFormData(fd, triggerBtn) {
      triggerBtn.disabled = true;
      statusBox.textContent = "Running inference...";
      resultsBox.innerHTML = "";

      try {
        const res = await fetch("/predict-both", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || "Request failed");
        }
        statusBox.textContent = "Done.";
        resultsBox.innerHTML =
          renderModelCard("Least Squares", data.least_squares) +
          renderModelCard("ResNet", data.resnet);
      } catch (err) {
        statusBox.innerHTML = `<span class="err">${err.message}</span>`;
      } finally {
        triggerBtn.disabled = false;
      }
    }

    function canvasToBlob() {
      return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob), "image/png");
      });
    }

    predictCanvasBtn.addEventListener("click", async () => {
      const blob = await canvasToBlob();
      if (!blob) return;
      preview.src = URL.createObjectURL(blob);
      previewWrap.style.display = "block";
      const fd = new FormData();
      fd.append("file", blob, "drawing.png");
      await runPredictWithFormData(fd, predictCanvasBtn);
    });

    clearCanvasBtn.addEventListener("click", () => {
      resetCanvas();
      preview.src = "";
      previewWrap.style.display = "none";
      resultsBox.innerHTML = "";
      statusBox.textContent = "Canvas cleared.";
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      await runPredictWithFormData(fd, submitUploadBtn);
    });
  </script>
</body>
</html>
"""


async def _read_upload_image(file: UploadFile) -> Image.Image:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG or JPG/JPEG.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: Literal["resnet", "least_squares"] = Form("resnet"),
    invert: bool = Form(False),
) -> dict:
    image = await _read_upload_image(file)

    try:
        result = service.predict(image=image, model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if invert:
        # Re-run with inverted image for white digit on black background scenarios.
        from PIL import ImageOps

        result = service.predict(image=ImageOps.invert(image.convert("L")), model_name=model_name)

    return result


@app.post("/predict-both")
async def predict_both(
    file: UploadFile = File(...),
    invert: bool = Form(False),
) -> dict:
    image = await _read_upload_image(file)

    if invert:
        from PIL import ImageOps

        image = ImageOps.invert(image.convert("L"))

    response: dict = {"least_squares": None, "resnet": None}
    for model_key in ("least_squares", "resnet"):
        try:
            response[model_key] = service.predict(image=image, model_name=model_key)
        except Exception as exc:  # keeps demo robust if one model is missing
            response[model_key] = {"error": str(exc)}
    return response
