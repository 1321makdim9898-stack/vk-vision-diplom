````md
# VK Vision — Product (React + TS + Express + ML service)

Основной “продуктовый” прототип находится в `apps/product/`.

## Архитектура
- `frontend/` — React + TypeScript (Vite)
- `backend/` — Node.js (Express): загрузка изображения, хранение истории (JSON), проксирование в ML-сервис
- `ml_service/` — FastAPI/uvicorn: детекция лиц (MediaPipe) + инференс (PyTorch)

---

## Требования
- Node.js 18+ (лучше 20+)
- Python 3.10–3.12 (желательно)
- (опционально) CUDA + совместимые версии torch/torchvision, если хочешь GPU

---

## Веса моделей (weights)
Веса (`*.pth`) **не хранятся в git**.

### Куда класть
Рекомендуемый вариант — папка `models/` в корне репозитория.

Создай папку в корне репозитория:
- `models/`

### Ожидаемые имена файлов (важно)
ML-сервис ищет веса строго по именам:

- `models/emotion_resnet18.pth` *(эмоции, ResNet18)*
- `models/age_gender_dldl_resnet50.pth` *(возраст/пол, DLDL, ResNet50 — приоритет №1)*
- `models/age_gender_regression_resnet18.pth` *(fallback №2)*
- `models/age_gender_resnet18.pth` *(fallback №3, возраст по бинам)*

> Возраст/пол выбирается по приоритету: **DLDL → regression → bins**.

### Переменная окружения для пути к моделям
Если веса лежат не в `./models`, можно указать:
- `MODELS_DIR` — путь к папке с весами (например абсолютный путь)

---

## Запуск (Windows, PowerShell)

Запускать **в 3 отдельных терминалах**: ML → Backend → Frontend.

### 1) ML service (Python, порт 8001)

```powershell
cd apps/product/ml_service

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Если models лежит в корне репозитория — можно не задавать MODELS_DIR.
# Если лежит где-то ещё:
# $env:MODELS_DIR="C:\path\to\models"

uvicorn app:app --host 127.0.0.1 --port 8001
````

### 2) Backend (Node/Express, порт 3001)

Backend по умолчанию поднимается на `127.0.0.1:3001` и:

* раздаёт загруженные файлы по `/uploads`
* имеет health-check `/api/health`
* отправляет изображение в ML по `POST {ML_SERVICE_URL}/infer`

Запуск:

```powershell
cd apps/product/backend
npm i

$env:ML_SERVICE_URL="http://127.0.0.1:8001"
# $env:UPLOADS_DIR="C:\vk-vision-data\uploads"
# $env:STORAGE_DIR="C:\vk-vision-data\storage"

npm run dev
```

### 3) Frontend (Vite dev server, порт 5173)

Запуск:

```powershell
cd apps/product/frontend
npm i
npm run dev
```

Открыть в браузере:

* `http://127.0.0.1:5173`

> В dev-режиме фронт отправляет запросы в backend через прокси `/api`.

---

## Проверка работоспособности

### ML service

* `GET http://127.0.0.1:8001/health` — статус зависимостей и загрузки весов
* `POST http://127.0.0.1:8001/infer` — инференс (multipart/form-data, поле `image` или `file`)

### Backend

* `GET http://127.0.0.1:3001/api/health` — health-check backend + текущий `ML_SERVICE_URL`
* `GET http://127.0.0.1:3001/api/history?limit=20` — история (limit 1..200)

### Frontend

* открыть `http://127.0.0.1:5173`
* в dev-режиме фронт ходит в backend через прокси `/api`

---

## Переменные окружения

### Backend

* `ML_SERVICE_URL` (default: `http://127.0.0.1:8001`) — куда отправлять `/infer`
* `HOST` (default: `127.0.0.1`) — хост сервера
* `PORT` (default: `3001`) — порт сервера
* `UPLOADS_DIR` — папка для загруженных файлов (default: `apps/product/backend/uploads`)
* `STORAGE_DIR` — папка для истории (default: `apps/product/backend/src/storage`)

### ML service

* `MODELS_DIR` — путь к папке с `.pth` (если не используешь `./models`)

---

## API

### Backend

* `POST /api/analyze` — multipart/form-data, поле файла: `file` или `image`
* `GET /api/history?limit=20`
* `GET /api/health`
* `GET /uploads/<filename>` — статическая раздача загруженных изображений

### ML service

* `GET /health`
* `POST /infer` — multipart/form-data, поле: `image` или `file`

```
::contentReference[oaicite:0]{index=0}
```
