
import random
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/classifiers", tags=["Classifiers"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str | None = Form(default=None),
):
    # Validate the uploaded file extension
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, or PNG image.",
        )

    # Check that the file is not empty
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    # MOCK ONLY: replace this section with real model inference later
    label = random.choice(["NORMAL", "PNEUMONIA"])
    probability = round(random.uniform(0.70, 0.99), 3)

    return {
        "patient_id": patient_id,
        "filename": file.filename,
        "prediction": label,
        "probability": probability,
        "is_mock": True,
        "message": (
            "Demo result only. This is a random prediction, "
            "not a medical assessment."
        ),
    }
