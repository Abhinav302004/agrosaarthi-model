import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, List
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; replace with your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get model directory
def _get_model_dir():
    if "MODEL_DIR" not in os.environ:
        raise Exception("MODEL_DIR environment variable is not set.")
    return os.environ["MODEL_DIR"]

model = None
pesticide_mapping = None

# Load model and pesticide mapping
def load_model():
    global model, pesticide_mapping
    model_dir = _get_model_dir()
    model_path = os.path.join(model_dir, "NEW_IP102_BEST.pt")  # Updated to match uploaded file

    try:
        model = YOLO(model_path)
        logger.info(f"✅ YOLO model loaded from: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise

    mapping_path = os.path.join(model_dir, "Final_Pesticides.csv")  # Updated to match uploaded file
    try:
        pesticide_mapping = pd.read_csv(mapping_path)
        pesticide_mapping['Pest Name'] = pesticide_mapping['Pest Name'].str.lower().str.strip()
        logger.info(f"✅ Pesticide mapping loaded from: {mapping_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load pesticide mapping: {e}")
        raise

    return model

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = _get_model_dir()
    logger.info(f"📂 MODEL_DIR = {model_dir}")
    logger.info(f"📁 Files in model dir: {os.listdir(model_dir)}")
    logger.info("🚀 Starting AgroSaarthi API...")
    load_model()
    yield
    logger.info("🛑 Shutting down AgroSaarthi API...")
    


# FastAPI app instance
app = FastAPI(
    lifespan=lifespan,
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", "")
)

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "AgroSaarthi YOLO API is running"}

# Health check endpoint
@app.get("/health")
async def health() -> Dict[str, bool]:
    return {"healthy": True}

# Pydantic request model
class PredictRequest(BaseModel):
    description: str = Field(
        "Upload an image file of pest-infected crop for diagnosis.",
        example="pest-image.jpg"
    )

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, pesticide_mapping
    if model is None or pesticide_mapping is None:
        logger.error("Model or pesticide mapping not loaded.")
        raise HTTPException(status_code=500, detail="Model or pesticide mapping not loaded")

    logger.info(f"📥 Received file: {file.filename}")
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("❌ Invalid image file.")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    logger.info("🔎 Running YOLO inference...")
    results = model.predict(image, conf=0.5)
    detections = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        confidence = float(result.conf[0])
        class_id = int(result.cls[0])
        class_name = model.names[class_id].lower().strip()

        logger.info(f"✅ Detected: {class_name} (conf: {confidence:.2f})")

        match = pesticide_mapping[pesticide_mapping['Pest Name'] == class_name]
        if not match.empty:
            row = match.iloc[0]
            pesticide_info = {
                "pesticides": row["Pesticides"],
                "crop": row["Crop"],
                "type": row["Pesticide Type"]
            }
        else:
            pesticide_info = {
                "pesticides": "Not available",
                "crop": "Unknown",
                "type": "Unknown"
            }

        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "class_name": class_name,
            **pesticide_info
        })

    logger.info(f"📤 Returning {len(detections)} detections.")
    return {"detections": detections}