import os
import uuid
from pathlib import Path

from flask import Flask, request, render_template, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from rag_assistant import KnowledgeBase, build_explanation

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret-key")

CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
knowledge_base = KnowledgeBase(KNOWLEDGE_DIR)


def resolve_model_path():
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    default_path = BASE_DIR / "models" / "model_inception.h5"
    if default_path.exists():
        return default_path

    legacy_path = BASE_DIR.parent / "Computer Vision" / "Transfer Learning" / "model_inception.h5"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        "Model file not found. Set MODEL_PATH or place model in models/model_inception.h5"
    )


MODEL_PATH = resolve_model_path()
model = load_model(str(MODEL_PATH))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(229, 229))
    x = img_to_array(img).astype("float32")
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_probs(image_path):
    x = preprocess_image(image_path)
    preds = model.predict(x, verbose=0)[0]
    preds = preds.astype(float)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return label, confidence, probs


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please choose an image file.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a JPG or PNG.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = UPLOAD_DIR / filename
        file.save(filepath)

        try:
            label, confidence, probs = predict_probs(filepath)
        except Exception:
            flash("Model inference failed. Please try another image.")
            return redirect(request.url)

        probs_list = [(name, probs[name]) for name in CLASS_NAMES]
        explanation = build_explanation(label, confidence, probs, knowledge_base)
        return render_template(
            "index.html",
            filename=filename,
            label=label,
            confidence=f"{confidence * 100:.2f}%",
            probs=probs_list,
            explanation=explanation,
        )

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "file is required"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    filename = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    label, confidence, probs = predict_probs(filepath)
    explanation = build_explanation(label, confidence, probs, knowledge_base)
    response = {
        "predicted_class": label,
        "confidence": confidence,
        "probabilities": {k.lower(): v for k, v in probs.items()},
        "model_version": MODEL_PATH.name,
        "explanation": explanation,
    }
    return jsonify(response)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
