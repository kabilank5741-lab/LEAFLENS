LEAFLENS SMARTVERIFY — UPDATED OFFLINE-FIRST PROTOTYPE
=====================================================

ADDED IN THIS VERSION
---------------------
1. Voice output in English/Tamil/Hindi using browser speech synthesis where available.
2. Voice commands: scan, history, guidance, listen, stop (browser support and microphone permission required).
3. Language selector: English, தமிழ், हिन्दी. Main controls/result guidance are translated; some long explanatory page copy remains English.
4. PWA shell caching (service worker) so the interface can reopen after it has been loaded once online.
5. On-device browser AI inference support using TensorFlow.js if a trained model exists at frontend/model/model.json and frontend/model/class_names.json.
6. Server-side model support using backend/model.keras + backend/class_names.json.
7. Uncertain cases are saved locally and posted to /api/expert-review when the server is reachable.
8. SQLite review queue endpoint. This stores cases on the configured server; it does NOT send a notification to an external expert until you configure an expert portal/email/service.
9. Removed random demo predictions from backend. Without a trained model, the app returns an honest model-not-installed message rather than fake AI results.

IMPORTANT MODEL LIMITATION
-------------------------
The ZIP does NOT include a labelled plant disease dataset or a trained disease model. So true AI prediction will not work until you train/export a model. The included train_model.py is the training pipeline, not a pretrained model. A 1–2 hour training window is not guaranteed; it depends on dataset size, CPU/GPU and package installation. Use a small, clearly scoped class set for a quick baseline, then test carefully.

RUN WEBSITE (Windows)
---------------------
1. Extract this ZIP.
2. Install Python 3.11 if needed.
3. Open backend folder and run start_windows.bat (or: python -m pip install -r requirements.txt then python -m uvicorn app:app --host 0.0.0.0 --port 8000).
4. Open http://127.0.0.1:8000 . For phone testing on the same Wi-Fi, open http://YOUR-COMPUTER-LAN-IP:8000 on phone. Allow the port through Windows Firewall if asked.
5. For true PWA install/offline caching, production deployment should use HTTPS. Localhost is treated as secure on the computer; a LAN HTTP URL may restrict service worker/microphone features on phones.

TRAIN A MODEL
-------------
Dataset layout:
 backend/dataset/train/<class_name>/*.jpg
 backend/dataset/validation/<class_name>/*.jpg

Install training dependencies in a separate environment:
 python -m pip install -r requirements-training.txt

Train a quick baseline (try 3–5 epochs first):
 python train_model.py --data dataset --epochs 5 --batch-size 16

For offline browser/phone inference export:
 python train_model.py --data dataset --epochs 5 --batch-size 16 --export-tfjs

This creates backend/model.keras + backend/class_names.json and frontend/model/model.json + weight files + class_names.json. Reload the app online once so the model files and TensorFlow.js library are cached. Confirm prediction still works after switching to airplane mode. If TFJS export fails due to dependency compatibility, use a compatible Python 3.10/3.11 environment and matching TensorFlow/TensorFlow.js versions.

DATASET
-------
Use a correctly labelled, legally usable plant disease dataset (for example, a selected subset of PlantVillage). Keep only the classes you can test. Folder names become the model labels. Dataset-only accuracy does not guarantee field accuracy. Include photos from real phones and varied lighting in evaluation.

EXPERT REVIEW
-------------
Uncertain cases are kept in localStorage while offline. When the configured server becomes reachable, the browser POSTs metadata to /api/expert-review. The server stores it in backend/leaflens_review.sqlite3. Review queue can be viewed at /api/expert-review. This is a queue only, not an actual expert notification. For a real deployment, host the API securely and connect a verified expert dashboard/contact workflow. Avoid sending identifiable farmer data without consent.

VOICE / LANGUAGES
-----------------
Voice input depends on browser support (typically Chromium-based browsers), microphone permission and speech recognition language availability. Voice output depends on installed device voices. Offline speech may not be available for every language/device. The app uses text guidance as fallback.

SAFETY
------
Prediction is triage, not a confirmed diagnosis. Confidence is not a guarantee of correctness. Do not prescribe pesticide dosage from this prototype. Get an agricultural expert to validate labels, thresholds and local-language advice before field use.


LANGUAGE + SPOKEN RESULT UPDATE
-------------------------------
The interface language selector supports English, Tamil, and Hindi. In the result panel, press Listen to hear the predicted class, confidence score, safety note, and next-step guidance in the selected language. Voice output uses the browser/device Speech Synthesis voices. Tamil/Hindi voice availability and offline speech depend on the operating system/browser and installed voices; if a voice is unavailable, the written result remains available. The voice assistant's listen command uses the same localized spoken result.
