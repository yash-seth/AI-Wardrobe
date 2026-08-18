from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .services.extractor import (
    PROJECT_ROOT,
    STATIC_DIR,
    DATA_DIR,
    UPLOADS_DIR,
    ensure_directories,
    load_metadata,
    process_image,
)

app = FastAPI(title="AI Wardrobe API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_directories()
app.mount("/wardrobe", StaticFiles(directory=str(DATA_DIR / "wardrobe")), name="wardrobe")
app.mount("/labelled_images", StaticFiles(directory=str(DATA_DIR / "labelled_images")), name="labelled_images")
app.mount("/Images", StaticFiles(directory=str(UPLOADS_DIR)), name="Images")

@app.get("/")
def home():
    ui_path = STATIC_DIR / "wardrobe-hanger-ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return JSONResponse({"message": "Upload UI file not found."}, status_code=404)

@app.get("/api/wardrobe")
def get_wardrobe():
    return {"items": load_metadata()}

@app.post("/api/process")
async def upload_and_process(
    file: UploadFile = File(...),
    save_tops: bool = Form(True),
    save_pants: bool = Form(True),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG, PNG, or WEBP.")

    unique_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    destination = UPLOADS_DIR / unique_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        return process_image(str(destination), save_tops=save_tops, save_pants=save_pants)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc