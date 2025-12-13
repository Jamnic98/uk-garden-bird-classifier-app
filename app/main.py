from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import onnxruntime as ort
import os

templates = Jinja2Templates(directory="templates")  # templates folder

app = FastAPI()

# Allow requests from frontend if served separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load local ONNX model ---
LOCAL_MODEL_PATH = os.path.join("models", "cnn_best_single.onnx")
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {LOCAL_MODEL_PATH}")

ort_session = ort.InferenceSession(LOCAL_MODEL_PATH)

# --- Class names ---
birds = [
    "blackbird",
    "blue tit",
    "carrion crow",
    "goldfinch",
    "great tit",
    "house sparrow",
    "magpie",
    "ring-necked parakeet",
    "robin",
    "starling",
    "wood pigeon",
]

# --- Preprocess PIL image for ONNX ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5           # normalize same as training
    img_array = np.transpose(img_array, (2, 0, 1)) # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    return img_array

# --- Predict using ONNX ---
def predict_class(image: Image.Image):
    input_tensor = preprocess_image(image)
    outputs = ort_session.run(None, {"input": input_tensor})
    pred_index = np.argmax(outputs[0], axis=1)[0]
    return birds[pred_index]

# --- FastAPI endpoints ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        return {"error": "Invalid image file."}

    prediction = predict_class(image)
    return {"prediction": prediction}

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", port=8080, log_level="debug", reload=True, access_log=False
    )
