# Deploy MNIST Demo (FastAPI + 2 Models)

This project serves:
- API docs: `/docs`
- Demo UI: `/demo`

## 1) Prepare project files

Make sure these files exist in your repo:
- `app.py`
- `inference.py`
- `requirements.txt`
- `Dockerfile`
- `models/resnet_mnist_state_dict.pt`
- `models/least_squares_W.npy`

## 2) Push to GitHub

```bash
cd "/Users/minhtiennguyen/Downloads/Mnist_Project"
git init
git add .
git commit -m "Initial MNIST demo deploy setup"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## 3) Deploy on Render (recommended)

1. Go to [Render](https://render.com/) and create **Web Service** from your GitHub repo.
2. Environment:
   - Runtime: `Docker`
   - Instance: Free/Starter (your choice)
3. Add environment variables (optional, only if custom model paths):
   - `RESNET_PATH=/app/models/resnet_mnist_state_dict.pt`
   - `LS_PATH=/app/models/least_squares_W.npy`
4. Deploy.

After success:
- `https://<your-service>.onrender.com/demo`
- `https://<your-service>.onrender.com/docs`

## 4) If model file is too large

If GitHub blocks large model files:
- Option A: Git LFS
- Option B: Download models from cloud storage at startup

For portfolio simplicity, start with smaller model artifacts or Git LFS.

## 5) Portfolio links

Put these links in README / CV:
- Live demo: `.../demo`
- API docs: `.../docs`
- Source code: GitHub repo
