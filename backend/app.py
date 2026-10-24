from pathlib import Path
from datetime import datetime, timezone
import io, json, sqlite3, re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_PATH = FRONTEND_DIR / "model" / "leaflens_cotton_model.tflite"
LABELS_PATH = FRONTEND_DIR / "model" / "class_names.json"
DB_PATH = APP_DIR / "leaflens_review.sqlite3"
MAX_FILE_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
app = FastAPI(title="LeafLens SmartVerify API", description="Offline-first crop leaf triage prototype", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
_model = None
_labels = None

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS expert_reviews (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL, received_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending')")
    con.commit()
    return con

def validate_image(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)); image.verify()
        image = Image.open(io.BytesIO(image_bytes)); width, height = image.size
        if width < 80 or height < 80:
            raise HTTPException(status_code=400, detail="Image is too small. Please upload a clear photo at least 80 × 80 pixels.")
        return {"width": width, "height": height, "format": image.format}
    except HTTPException: raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Could not read this image. Try a JPG, PNG, or WEBP file.")

def load_model():
    global _model, _labels

    if _model is not None and _labels is not None:
        return _model, _labels

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("TFLite model or class_names.json not found.")
        return None, None

    try:
        import tensorflow as tf

        _model = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        _model.allocate_tensors()

        _labels = json.loads(
            LABELS_PATH.read_text(encoding="utf-8")
        )

        print("TFLite model loaded successfully.")
        print("Classes:", _labels)

        return _model, _labels

    except Exception as exc:
        print(f"Model load failed: {exc}")
        _model = None
        _labels = None
        return None, None

def predict_with_model(image_bytes: bytes):
    interpreter, labels = load_model()

    if interpreter is None:
        return None

    import numpy as np

    try:
        image = (
            Image.open(io.BytesIO(image_bytes))
            .convert("RGB")
            .resize((160, 160))
        )

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_info = input_details[0]
        output_info = output_details[0]

        arr = np.asarray(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        if input_info["dtype"] == np.uint8:
            arr = arr.astype(np.uint8)
        elif input_info["dtype"] == np.int8:
            scale, zero_point = input_info["quantization"]
            if scale:
                arr = np.round(arr / scale + zero_point)
            arr = np.clip(arr, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_info["index"], arr)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_info["index"])[0]
        scores = np.asarray(scores, dtype=np.float32)

        # Convert logits to probabilities if necessary.
        if (
            np.any(scores < 0)
            or np.any(scores > 1)
            or not np.isclose(np.sum(scores), 1.0, atol=0.05)
        ):
            exp_scores = np.exp(scores - np.max(scores))
            scores = exp_scores / np.sum(exp_scores)

        order = np.argsort(scores)[::-1][:3]

        matches = [
            {
                "label": labels[int(i)]
                if int(i) < len(labels)
                else f"Class {int(i)}",
                "score_percent": round(float(scores[i]) * 100, 1),
            }
            for i in order
        ]

        top = matches[0]
        confidence = top["score_percent"]

        return {
            "prediction_mode": "server_ai",
            "is_real_ai_prediction": True,
            "label": top["label"],
            "confidence_percent": confidence,
            "status": "needs_verification" if confidence >= 60 else "uncertain",
            "top_matches": matches,
            "advice": [
                "Take another clear photo in natural daylight.",
                "Check nearby leaves and record whether symptoms are spreading.",
                "Do not apply pesticides based only on an app prediction; verify with a local agricultural expert.",
            ],
            "safety_message": (
                "AI suggestion only, not a confirmed diagnosis. "
                "Verify with an agricultural expert before treatment."
            ),
        }

    except Exception as exc:
        print(f"TFLite prediction failed: {exc}")
        raise

class ReviewCase(BaseModel):
    case_id: str
    crop: str = "Other"
    label: str
    confidence_percent: float = 0
    status: str = "uncertain"
    created_at: str = ""
    notes: str = ""
    prediction_mode: str = "unknown"

@app.get('/api/health')
def health():
    model, labels = load_model()
    return {"status":"ok","service":"LeafLens SmartVerify API","time_utc":datetime.now(timezone.utc).isoformat(),"model_status":"trained_model_loaded" if model is not None else "model_not_installed","review_queue":"local_sqlite_queue"}

@app.post('/api/analyze')
async def analyze_leaf(image: UploadFile = File(...), crop: str = Form('Other'), notes: str = Form(''), language: str = Form('en')):
    if image.content_type not in ALLOWED_TYPES: raise HTTPException(status_code=415, detail="Unsupported file type. Use JPG, PNG, or WEBP.")
    image_bytes = await image.read()
    if not image_bytes: raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_FILE_BYTES: raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 8 MB.")
    metadata = validate_image(image_bytes)
    result = predict_with_model(image_bytes)
    if result is None: raise HTTPException(status_code=503, detail="No trained server model is installed. Install model.keras and class_names.json, or use the offline browser model in frontend/model/.")
    return {"case_id":"LL-"+datetime.now(timezone.utc).strftime('%y%m%d%H%M%S%f'),"created_at":datetime.now(timezone.utc).isoformat(),"crop":crop,"notes_received":bool(notes.strip()),"notes":notes,"image":metadata,"image_quality":{"status":"basic_dimensions_checked","message":"Image file decoded and dimensions checked; blur is not assessed by this version."},**result}

@app.post('/api/expert-review')
async def submit_expert_review(payload: str = Form(...), image: UploadFile | None = File(None)):
    """Queue case metadata and optional leaf photo on this server. No external expert is notified automatically."""
    try: item = ReviewCase.model_validate_json(payload)
    except Exception: raise HTTPException(status_code=400, detail='Invalid review case payload.')
    photo_path = None
    if image is not None:
        if image.content_type not in ALLOWED_TYPES: raise HTTPException(status_code=415, detail='Unsupported review photo type.')
        image_bytes = await image.read()
        if not image_bytes or len(image_bytes) > MAX_FILE_BYTES: raise HTTPException(status_code=400, detail='Review photo is empty or exceeds 8 MB.')
        validate_image(image_bytes)
        review_dir = APP_DIR / 'review_uploads'; review_dir.mkdir(exist_ok=True)
        safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', item.case_id)[:80]
        suffix = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp'}[image.content_type]
        photo_file = review_dir / f'{safe_id}{suffix}'; photo_file.write_bytes(image_bytes)
        photo_path = str(photo_file.relative_to(APP_DIR))
    record = item.model_dump(); record['photo_path'] = photo_path
    con = db()
    try:
        con.execute('INSERT OR REPLACE INTO expert_reviews(case_id,payload,received_at,status) VALUES(?,?,?,?)', (item.case_id, json.dumps(record), datetime.now(timezone.utc).isoformat(), 'pending'))
        con.commit()
    finally: con.close()
    return {"status":"queued","case_id":item.case_id,"photo_received":photo_path is not None,"message":"Stored in this server review queue. Configure an expert dashboard/notification to reach an actual expert."}

@app.get('/api/expert-review')
def list_expert_reviews():
    con = db()
    try: rows = con.execute('SELECT case_id,payload,received_at,status FROM expert_reviews ORDER BY received_at DESC').fetchall()
    finally: con.close()
    return {"count":len(rows),"items":[{"case_id":r['case_id'],"payload":json.loads(r['payload']),"received_at":r['received_at'],"status":r['status']} for r in rows]}

if FRONTEND_DIR.exists() and (FRONTEND_DIR / 'index.html').exists():
    app.mount('/', StaticFiles(directory=str(FRONTEND_DIR), html=True), name='frontend')
