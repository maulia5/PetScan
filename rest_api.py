import os
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from insightface.app import FaceAnalysis

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
UPLOAD_TRAIN_FOLDER = "static/train"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== CONFIG ==================
UNKNOWN_THR = 0.38
PACK_NAME = "buffalo_s"
DET_SIZE = (448, 448)

CENT_PATH = Path("models/face_db_centroids.pkl")
EMB_PATH = Path("models/embeddings_buffalo_s.pkl")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ================== LOAD DB =================
def load_model_centroids():
    # 1) coba load dari face_db_centroids.pkl
    if CENT_PATH.exists():
        db = pickle.loads(CENT_PATH.read_bytes())
        labels = np.array(db["labels"], dtype=object)
        cents = db["centroids"].astype(np.float32)
        print(f"✅ Loaded centroids: {len(labels)} persons")
        return labels, cents

    # 2) fallback: bangun centroid dari embeddings_buffalo_s.pkl
    if EMB_PATH.exists():
        db = pickle.loads(EMB_PATH.read_bytes())
        names = np.array(db["names"], dtype=object)
        embs = db["embs"].astype(np.float32)

        uniq_labels = sorted(set(names.tolist()))
        cents = []
        for lab in uniq_labels:
            V = embs[names == lab]
            cents.append(l2n(V.mean(0)))
        cents = np.vstack(cents).astype(np.float32)
        print(f"⚠️ Built centroids from {EMB_PATH.name}: {len(uniq_labels)} persons")
        return np.array(uniq_labels, dtype=object), cents

    raise SystemExit(
        "❌ Tidak ada 'models/face_db_centroids.pkl' maupun "
        "'embeddings_buffalo_s.pkl'. Jalankan script training / build_embeddings dulu."
    )


def make_app():
    app = FaceAnalysis(name=PACK_NAME)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    det = app.models.get("detection")
    if hasattr(det, "threshold"):
        det.threshold = 0.22  # bisa kamu tweak
    return app


# ================== UTIL ====================
def l2n(v, eps=1e-12):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + eps)


def cosine_sim(a, b):
    # a: (D,), b: (N,D)
    return a @ b.T


# ================== PREDIKSI =================
def predict_image(img_path: Path, app, labels, cents):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Gagal baca gambar: {img_path}")
        return None, []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    preds = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        emb = l2n(f.embedding)
        sims = cosine_sim(emb, cents)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        name = labels[idx] if score >= UNKNOWN_THR else "Unknown"
        preds.append((name, score, (x1, y1, x2, y2)))

    return img, preds


def annotate(img, preds):
    im = img.copy()
    if not preds:
        cv2.putText(
            im,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return im

    for name, score, (x1, y1, x2, y2) in preds:
        is_unk = name == "Unknown"
        color = (60, 60, 230) if is_unk else (80, 220, 80)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({score * 100:.1f}%)"
        cv2.putText(
            im,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return im

@app.get("/tes")
def tes():
    return jsonify({"tes":"tes"})

@app.route("/attendance/register", methods=["POST"])
def attendance_register():
    if request.method == "POST":
        if not request.files.get("image") is not None:
            return jsonify(
                {"status": "INVALID_FILE", "message": "Filename is required"}
            ), 200
        file = request.files["image"]
        if file.filename == "":
            return jsonify(
                {"status": "INVALID_FILE", "message": "Filename is required"}
            )

        allowed_extensions = ["jpg", "jpeg", "png"]
        extension = file.filename.rsplit(".", 1)[1].lower()

        if extension not in allowed_extensions:
            return jsonify(
                {
                    "status": "INVALID_FILE",
                    "message": "Only jpg, jpeg and png are allowed",
                }
            )
        if not "email" in request.form:
            return jsonify( {
                    "status": "INVALID_EMAIL",
                    "message": "Email is required",
                })
        email = request.form["email"]
        if not email:
            return jsonify(
                {"status": "INVALID_EMAIL", "message": "Email is required"}
            )
        if len(email) <= 0:
            return jsonify({"status": "INVALID_EMAIL", "message": "Email is required"})

        filename = f"{email}_train_01.{extension}"
        file_path = os.path.join(UPLOAD_TRAIN_FOLDER, filename)
        file.save(file_path)
        subprocess.Popen(["./train_test_and_eval.sh"], stdin=subprocess.PIPE)
        return jsonify({"status": "OK", "filename": filename})
    else:
        return jsonify({"satus": "INVALID", "error": "Method not allowed"}), 405


@app.route("/attendance/predict", methods=["POST"])
def attendance_predict():
    if request.method == "POST":
        if request.files.get("face") is  None:
            return jsonify(
                {"status": "INVALID_FILE", "label": None, "message": "Filename is required"}
            ), 200

        file = request.files["face"]
        if file.filename == "":
            return jsonify(
                {"status": "INVALID_FILE", "label": None, "message": "Filename is required"}
            ),200

        allowed_extensions = ["jpg", "jpeg", "png"]
        extension = file.filename.rsplit(".", 1)[1].lower()

        if extension not in allowed_extensions:
            return jsonify(
                {
                    "status": "INVALID_FILE",
                    "label": None,
                    "message": "Only jpg, jpeg and png are allowed",
                }
            ),200
        if  not "email" in request.form:
            return jsonify({"status": "INVALID_EMAIL","label":None, "message": "Email is required"}),200

        email = request.form["email"]
        if not email:
            return jsonify({"status": "INVALID_EMAIL", "label": None, "message": "Email is required"}),200
        if len(email) <= 0:
            return jsonify({"status": "INVALID_EMAIL", "label": None, "message": "Email is required"}),200

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # load DB & model sekali di awal
        labels, cents = load_model_centroids()
        predapp = make_app()

        img_path = Path(file_path)
        img, preds = predict_image(img_path, predapp, labels, cents)
        os.remove(file_path)
        if len(preds) <= 0:
            return jsonify(
                {"status": "INVALID_FILE","label": None, "message": "No face detected"}
            ), 200
        label = preds[0]
        if len(preds) <= 0:
            return jsonify({"status": "INVALID_FILE","label": None, "message": "No face detected"}),200
        result = label[0]
        if len(result) <= 0:
            return jsonify({"status": "INVALID_FILE","label": None, "message": "No face detected"}),200

        if email.lower().__eq__(result.lower()):
            return jsonify({"status": "MATCH","label": result.lower(), "message": "Face matched"}),200

        return jsonify({"status": "UNMATCH", "label": result.lower(), "message": "success"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
