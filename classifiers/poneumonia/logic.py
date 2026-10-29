import io
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/classifiers", tags=["Classifiers"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_PATH = Path(__file__).resolve().parent / "models" / "resnet50.pt"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_MODEL: Optional[nn.Module] = None


def _build_resnet50_from_state_dict(state_dict: Dict[str, Any]) -> nn.Module:
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = models.resnet50(weights=None)

    if "fc.weight" in clean_sd:
        out_features = clean_sd["fc.weight"].shape[0]
        in_features = clean_sd["fc.weight"].shape[1]
        model.fc = nn.Linear(in_features, out_features)
    elif "fc.1.weight" in clean_sd:
        out_features = clean_sd["fc.1.weight"].shape[0]
        in_features = clean_sd["fc.1.weight"].shape[1]
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, out_features),
        )

    model.load_state_dict(clean_sd, strict=False)
    return model


def get_model() -> nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    try:
        loaded = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
        loaded.eval()
        _MODEL = loaded
        return _MODEL
    except Exception:
        pass

    checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model = _build_resnet50_from_state_dict(state_dict)
    else:
        raise RuntimeError("Unsupported checkpoint format in resnet50.pt")

    model.to(DEVICE)
    model.eval()
    _MODEL = model
    return _MODEL


def run_inference(image_bytes: bytes) -> tuple[str, float]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.") from exc

    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    model = get_model()

    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if outputs.ndim == 1 or outputs.shape[-1] == 1:
            prob_pneumonia = float(torch.sigmoid(outputs.view(-1))[0].item())
            if prob_pneumonia >= 0.5:
                return "PNEUMONIA", round(prob_pneumonia, 4)
            return "NORMAL", round(1.0 - prob_pneumonia, 4)

        probs = torch.softmax(outputs, dim=-1)[0]
        conf, idx = torch.max(probs, dim=0)
        label_idx = int(idx.item())
        label = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"CLASS_{label_idx}"
        return label, round(float(conf.item()), 4)


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str | None = Form(default=None),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, or PNG image.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    try:
        label, probability = run_inference(content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(exc)}",
        ) from exc

    return {
        "patient_id": patient_id,
        "filename": file.filename,
        "prediction": label,
        "probability": probability,
        "is_mock": False,
        "message": "ResNet50 inference completed. Decision support only, not a standalone diagnosis.",
    }
