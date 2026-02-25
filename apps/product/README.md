# VK Vision — Product Prototype (React+TS + Express(CommonJS) + ML Service)

## Что внутри
- `frontend/` — React + TypeScript (Vite)
- `backend/` — Express (CommonJS), загрузка файлов, история анализов (JSON)
- `ml_service/` — FastAPI сервис `/infer` (попытка использовать ваш `ml_api.py`)

## Запуск (Windows, PowerShell)

### 1) ML сервис (Python)
```powershell
cd C:\vk-vision-demo\ml_service
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# ВАЖНО: чтобы импортировался ваш ml_api.py:
set PROJECT_ROOT=C:\vk-vision-demo
# Модели:
set EMOTION_MODEL=models\emotion_resnet18.pth
set AGE_GENDER_MODEL=models\age_gender_resnet18.pth

uvicorn app:app --host 127.0.0.1 --port 8001
```

### 2) Backend (Node.js)
```powershell
cd C:\vk-vision-demo\backend
copy .env.example .env
npm i
npm run dev
```

### 3) Frontend (React)
```powershell
cd C:\vk-vision-demo\frontend
npm i
npm run dev
```

Открыть: http://127.0.0.1:5173

## API
- POST `/api/analyze` (multipart/form-data, поле `image`)
- GET `/api/history?limit=20`
- `/uploads/<filename>` — раздача загруженных картинок
