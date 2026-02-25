# ml_api.py
"""
Простой HTTP-API для анализа изображений.
Использует уже готовый pipeline из demo_gradio.py
и возвращает JSON с результатами.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException

# Импортируем готовый пайплайн и структуры из твоего демо
from demo_gradio import pipeline  # AnalyzerPipeline с методами analyze

app = FastAPI(title="VK Visual Analyzer ML API")


def run_analysis_on_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Вспомогательная функция: байты -> BGR -> pipeline.analyze() -> JSON."""
    # Декодируем байты в изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Не удалось декодировать изображение")

    # Запускаем твой пайплайн
    result = pipeline.analyze(bgr)

    # Собираем JSON так же, как в analyze_image в demo_gradio.py
    faces_json: List[Dict[str, Any]] = []
    for fr in result.faces:
        faces_json.append(
            {
                "box": asdict(fr.box),
                "emotion": fr.emotion,
                "aus": fr.aus,
                "age_group": fr.age_group,
                "gender": fr.gender,
            }
        )

    if result.body is None:
        body_json: Optional[Dict[str, Any]] = None
    else:
        body_json = {
            "category": result.body.category,
            "scores": result.body.scores,
        }

    return {
        "faces": faces_json,
        "body": body_json,
        "meta": result.meta,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Простой healthcheck для проверки, что сервис жив."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image_api(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Принимает одно изображение (jpg/png) и возвращает JSON c результатом анализа.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Файл пустой")

        result_json = run_analysis_on_bytes(image_bytes)
        return result_json
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Логируем на всякий случай
        print("[ERROR] analyze_image_api:", e)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка анализа")
