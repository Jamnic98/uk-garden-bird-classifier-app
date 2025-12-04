from mangum import Mangum
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io

templates = Jinja2Templates(directory="app/templates")  # point to templates folder

app = FastAPI()

def handler(event: dict, _context=None):
    asgi_handler = Mangum(app)
    return asgi_handler(event, _context)

# Allow requests from frontend if served separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy predict function
def predict_class(image: Image.Image) -> str:
    # Replace with your model inference
    return "Carrion Crow"

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
