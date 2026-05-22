from __future__ import annotations

import atexit
import base64
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import zipfile
from typing import Optional

import cv2
import numpy as np
import torch

from flask import Flask, jsonify, render_template, request

from Major_Project import DEVANAGARI_CHARS, HybridCNNViT_HindiOCR, WebcamPreprocessor


app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_lock = threading.Lock()
_model = None
_preprocessor = None
_model_load_error = None
_temp_checkpoint_file = None


@app.get("/")
def home():
	return render_template("index.html")


@app.get("/health")
def health():
	ensure_model_loaded()
	return jsonify(
		{
			"ok": _model is not None,
			"service": "hindi-ocr-minimal",
			"device": DEVICE,
			"model_loaded": _model is not None,
			"model_error": _model_load_error,
		}
	)


def _safe_extract_base64(image_data: str) -> bytes:
	"""Accept data URLs or raw base64 and return decoded bytes."""
	if not image_data:
		return b""

	if "," in image_data and image_data.startswith("data:"):
		image_data = image_data.split(",", 1)[1]

	try:
		return base64.b64decode(image_data, validate=True)
	except Exception:
		return b""


def _pack_checkpoint_dir_to_torch_zip(source_dir: Path) -> str:
	"""
	Convert extracted torch archive dir -> temporary .pt zip file.
	Needed because current artifact is stored as folder: major_project_trained_model.keras/final_try/
	"""
	global _temp_checkpoint_file

	if _temp_checkpoint_file and Path(_temp_checkpoint_file).exists():
		return _temp_checkpoint_file

	with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
		tmp_path = tmp.name

	with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
		for item in source_dir.rglob("*"):
			if item.is_file():
				rel = str(item.relative_to(source_dir)).replace("\\", "/")
				zf.write(item, arcname=f"archive/{rel}")

	_temp_checkpoint_file = tmp_path
	return tmp_path


def _resolve_checkpoint_path() -> str:
	"""Support both file checkpoint and extracted checkpoint folder formats."""
	raw_path = Path("major_project_trained_model.keras")
	if raw_path.is_file():
		return str(raw_path)

	extracted = raw_path / "final_try"
	if extracted.is_dir():
		return _pack_checkpoint_dir_to_torch_zip(extracted)

	raise FileNotFoundError(
		"Checkpoint not found. Expected file 'major_project_trained_model.keras' "
		"or folder 'major_project_trained_model.keras/final_try'."
	)


def ensure_model_loaded():
	global _model, _preprocessor, _model_load_error

	if _model is not None:
		return

	with _model_lock:
		if _model is not None:
			return

		try:
			model = HybridCNNViT_HindiOCR(
				input_channels=1,
				feature_dim=256,
				embed_dim=384,
				vit_depth=6,
				vit_heads=6,
				num_classes=len(DEVANAGARI_CHARS),
				patch_size=4,
				pretrained_vit=True,
				char_list=DEVANAGARI_CHARS,
			).to(DEVICE)

			checkpoint_path = _resolve_checkpoint_path()
			checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
			model.load_state_dict(checkpoint["model_state_dict"])
			model.eval()

			_model = model
			_preprocessor = WebcamPreprocessor(target_size=(64, 256), device=DEVICE)
			_model_load_error = None
		except Exception as exc:
			_model = None
			_preprocessor = None
			_model_load_error = f"{type(exc).__name__}: {exc}"


def _decode_image_bytes_to_bgr(decoded: bytes):
	nparr = np.frombuffer(decoded, np.uint8)
	image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return image


def _build_frame_variants(roi: np.ndarray) -> list[np.ndarray]:
	"""Create lightweight variants to improve robustness under webcam lighting."""
	variants = [roi]

	bright = cv2.convertScaleAbs(roi, alpha=1.15, beta=8)
	dark = cv2.convertScaleAbs(roi, alpha=0.9, beta=-8)
	variants.extend([bright, dark])

	if min(roi.shape[:2]) < 80:
		upscaled = cv2.resize(roi, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
		variants.append(upscaled)

	return variants


def _predict_from_roi(roi: np.ndarray) -> tuple[str, float]:
	variants = _build_frame_variants(roi)
	tensors = [_preprocessor.preprocess_frame(v) for v in variants]
	batch = torch.cat(tensors, dim=0)

	with torch.no_grad():
		logits = _model(batch)
		texts, confs = _model.decode_predictions(logits)

	best_text = ""
	best_conf = 0.0
	for text, conf in zip(texts, confs):
		clean = (text or "").strip()
		c = float(conf or 0.0)
		if clean and c > best_conf:
			best_text = clean
			best_conf = c

	return best_text, best_conf


@atexit.register
def _cleanup_temp_checkpoint():
	global _temp_checkpoint_file
	try:
		if _temp_checkpoint_file and Path(_temp_checkpoint_file).exists():
			Path(_temp_checkpoint_file).unlink(missing_ok=True)
	except Exception:
		pass


@app.post("/predict")
def predict():
	payload = request.get_json(silent=True) or {}
	image_data = payload.get("file", "")
	decoded = _safe_extract_base64(image_data)

	if not decoded:
		return jsonify({"success": False, "error": "Invalid image payload"}), 400

	ensure_model_loaded()
	if _model is None or _preprocessor is None:
		return (
			jsonify(
				{
					"success": False,
					"error": "Model failed to load",
					"details": _model_load_error,
				}
			),
			503,
		)

	frame = _decode_image_bytes_to_bgr(decoded)
	if frame is None:
		return jsonify({"success": False, "error": "Invalid image format"}), 400

	try:
		roi, bbox = _preprocessor.extract_text_region(frame)
		if bbox is None:
			return jsonify(
				{
					"success": True,
					"text": "",
					"confidence": 0.0,
					"timestamp": datetime.now().strftime("%H:%M:%S"),
					"status": "no_text_region",
					"message": "No clear text area detected. Hold text in front of camera.",
				}
			)

		text, confidence = _predict_from_roi(roi)

		if confidence < 0.15:
			text = ""
	except Exception as exc:
		return jsonify({"success": False, "error": f"Inference error: {exc}"}), 500

	return jsonify(
		{
			"success": True,
			"text": text,
			"confidence": confidence,
			"timestamp": datetime.now().strftime("%H:%M:%S"),
			"status": "ok" if text else "low_confidence",
		}
	)


if __name__ == "__main__":
	import os

	port = int(os.getenv("PORT", "7860"))
	app.run(host="0.0.0.0", port=port, debug=False)

