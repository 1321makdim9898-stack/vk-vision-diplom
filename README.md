# VK Vision — Product Prototype (Client-Server)

This is a "product-like" version of the project:
- Backend: FastAPI (Python), stores analysis history in SQLite (can be upgraded to PostgreSQL)
- Frontend: simple HTML/JS single-page UI with cards + history
- ML: loads `models/emotion_resnet18.pth` and `models/age_gender_resnet18.pth` if present

## Run (Windows, PowerShell)

### 1) Backend
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Place trained models into:
- `backend/models/emotion_resnet18.pth`
- `backend/models/age_gender_resnet18.pth`  (optional; can be missing while training)

### 2) Frontend
Just open `frontend/index.html` in your browser.
(If CORS/file restrictions appear, you can serve it using a tiny server:)
```powershell
cd frontend
python -m http.server 5173
```
Then open: http://127.0.0.1:5173

## VK integration
We include best-effort endpoints:
- `/vk/resolve?screen_name=...&access_token=...`
- `/vk/profile_photos?owner_id=...&access_token=...`

Some methods may be unavailable depending on token/app type. This is expected and should be documented in the thesis.
