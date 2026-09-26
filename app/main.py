from __future__ import annotations
import io
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image, UnidentifiedImageError

from . import db
from .classifier import get_classifier

MODEL_VERSION = "rf-v2-eurosat7"  # bump when train_classifier.py output changes

app = FastAPI(title="GalaxEye Tile Classification Service (offline slice)")


@app.on_event("startup")
def startup():
    db.init_db()
    get_classifier()  # fail fast at boot if the model file is missing/corrupt


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tiles")
async def ingest_tile(file: UploadFile = File(...)):
    """Accept one tile image, classify it, persist the result."""
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(400, detail="File is not a readable image")

    clf = get_classifier()
    result = clf.predict(img)

    ingested_at = datetime.now(timezone.utc).isoformat()
    tile_id = db.insert_result(
        filename=file.filename or "unknown",
        ingested_at=ingested_at,
        result=result,
        model_version=MODEL_VERSION,
    )

    return {
        "id": tile_id,
        "filename": file.filename,
        "ingested_at": ingested_at,
        "model_version": MODEL_VERSION,
        **result,
    }


@app.get("/tiles")
def list_tiles(
    label: str | None = None,
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    low_confidence_only: bool = False,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
):
    """Analyst-facing query endpoint, e.g.
    GET /tiles?label=water&min_confidence=0.6
    GET /tiles?low_confidence_only=true   (tiles that need human review)
    """
    return db.query(
        label=label, min_confidence=min_confidence,
        low_confidence_only=low_confidence_only, limit=limit, offset=offset,
    )


@app.get("/tiles/{tile_id}")
def get_tile(tile_id: int):
    row = db.get_by_id(tile_id)
    if row is None:
        raise HTTPException(404, detail="Tile not found")
    return row


@app.get("/stats")
def get_stats():
    """Per-class counts and average confidence -- the cheapest offline
    'is the model drifting' signal, see Part 3 Q2."""
    return db.stats()
