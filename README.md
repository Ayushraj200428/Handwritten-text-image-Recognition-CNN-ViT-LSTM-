# AI-Powered Hindi OCR — Premium Web UI

This Flask web app serves a premium, dark-mode OCR dashboard for the
**Hybrid CNN + Vision Transformer** Hindi OCR model.

## What it does

- Uploads images (drag/drop, browse, camera upload).
- Sends the image to `/predict` and shows the recognized Hindi text.
- Displays confidence, timestamp, and AI processing status.

## Run locally

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Start server:

```powershell
python app.py
```

3. Open in browser:

```text
http://127.0.0.1:8000
```

## Deploy on Render (no extra files required)

1. Push this repo to GitHub.
2. Create a **Web Service** in Render.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

5. Ensure the model folder `major_project_trained_model.keras/` is included in the repo.

That’s it—Render will expose your public URL once the build completes.

