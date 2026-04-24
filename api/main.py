"""
Real Estate Image Classifier — FastAPI Backend
Model: ResNeXt101_32x8d optimized for F1 score
Swagger docs: http://localhost:8080/docs
Run with: uvicorn main:app --reload --port 8080
"""

from pathlib import Path
import io

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "resnext101_32x8d"
MODEL_PATH = Path(__file__).parent / "resnext101_32x8d_best_f1.pt"
IMAGE_SIZE = 224
DEVICE     = torch.device("mps" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial",
    "Inside city", "Kitchen", "Living room", "Mountain", "Office",
    "Open country", "Store", "Street", "Suburb", "Tall building",
]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model() -> nn.Module:
    model = torchvision.models.resnext101_32x8d(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base_model."):
            new_state_dict[k.replace("base_model.", "")] = v
        elif k.startswith("feature_extractor."):
            new_state_dict[k.replace("feature_extractor.", "")] = v
        elif k == "classifier.1.weight":
            new_state_dict["fc.weight"] = v
        elif k == "classifier.1.bias":
            new_state_dict["fc.bias"] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Real Estate Image Classifier",
    description=(
        "API of image classification for real estate properties in 15 categories.\n\n"
        "**Model:** ResNeXt101-32x8d with transfer learning.\n\n"
        "**Classes:** Bedroom, Coast, Forest, Highway, Industrial, Inside city, "
        "Kitchen, Living room, Mountain, Office, Open country, Store, Street, "
        "Suburb, Tall building."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: nn.Module = None


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. "
            "Copy resnext101_32x8d_best_f1.pt to the same folder as main.py."
        )
    model = load_model()
    print(f"Model '{MODEL_NAME}' loaded on {DEVICE}")


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    num_classes: int


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """Welcome and link to the documentation."""
    return {
        "message": "Real Estate Image Classifier API",
        "docs":    "http://localhost:8080/docs",
        "model":   MODEL_NAME,
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Checks that the API and model are operational."""
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=str(DEVICE),
        num_classes=len(CLASS_NAMES),
    )


@app.get("/classes", tags=["Info"])
def get_classes():
    """Returns the list of classes the model can predict."""
    return {"classes": CLASS_NAMES, "num_classes": len(CLASS_NAMES)}


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="JPG or PNG image of a real estate property"),
):
    """
    Classifies a real estate image into one of the 15 categories.

    - **file**: Image in JPG or PNG format.
    - **Returns**: Predicted class, confidence and probabilities for all classes.

    **Errors:**
    - `400`: The file is not a valid image.
    - `500`: Internal model error.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Format not supported: '{file.content_type}'. Use JPG or PNG.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the image. Please make sure the file is valid.",
        )

    try:
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs       = model(tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()

        predicted_idx   = int(torch.argmax(torch.tensor(probabilities)))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(probabilities[predicted_idx], 4)
        all_probs       = {cls: round(prob, 4) for cls, prob in zip(CLASS_NAMES, probabilities)}

        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")