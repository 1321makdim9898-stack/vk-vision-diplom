from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .db import init_db, insert_analysis, list_analyses, get_analysis
from .services.analyzer import Analyzer
from .integrations.vk.vk_client import VkClient

APP_NAME = "VK Vision Product Prototype"
DATA_DIR = Path("data")
THUMBS_DIR = DATA_DIR / "thumbs"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME)

# Allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for prototype; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
analyzer = Analyzer()

@app.get("/health")
def health():
    return {"status":"ok", "app": APP_NAME}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    source: str = Query("upload", description="upload | vk"),
    source_ref: Optional[str] = Query(None),
):
    if file.content_type not in ("image/jpeg","image/png","image/webp"):
        raise HTTPException(status_code=400, detail="Upload JPEG/PNG/WEBP image")

    raw = await file.read()
    result = analyzer.analyze(raw)

    # thumbnail
    thumb_bytes = analyzer.make_thumbnail(raw, size=260)
    ts = datetime.now(timezone.utc).isoformat()
    thumb_name = f"thumb_{int(datetime.now().timestamp()*1000)}.jpg"
    thumb_path = THUMBS_DIR / thumb_name
    thumb_path.write_bytes(thumb_bytes)

    rid = insert_analysis(
        created_at=ts,
        source=source,
        source_ref=source_ref,
        thumb_path=thumb_path.as_posix(),
        result_json=json.dumps(result, ensure_ascii=False),
    )

    return {"id": rid, "created_at": ts, "result": result}

@app.get("/history")
def history(limit: int = 25):
    items = list_analyses(limit=limit)
    # return only metadata; thumb is fetched separately
    return {"items":[
        {"id": it["id"], "created_at": it["created_at"], "source": it["source"], "source_ref": it["source_ref"]}
        for it in items
    ]}

@app.get("/history/{analysis_id}")
def history_item(analysis_id: int):
    it = get_analysis(analysis_id)
    if not it:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": it["id"],
        "created_at": it["created_at"],
        "source": it["source"],
        "source_ref": it["source_ref"],
        "result": json.loads(it["result_json"]),
    }

@app.get("/history/{analysis_id}/thumb")
def history_thumb(analysis_id: int):
    it = get_analysis(analysis_id)
    if not it:
        raise HTTPException(status_code=404, detail="Not found")
    p = it["thumb_path"]
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(p, media_type="image/jpeg")

# VK endpoints (best-effort for prototype)
@app.get("/vk/resolve")
def vk_resolve(screen_name: str, access_token: str):
    try:
        vk = VkClient(access_token)
        resp = vk.resolve_screen_name(screen_name.lstrip("@"))
        return {"response": resp}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/vk/profile_photos")
def vk_profile_photos(owner_id: int, access_token: str, count: int = 3):
    """
    NOTE: May fail depending on token scope/profile type.
    We keep endpoint to demonstrate integration and to support cases where it works.
    """
    try:
        vk = VkClient(access_token)
        resp = vk.get_profile_photos(owner_id, count=count)
        return {"response": resp}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
