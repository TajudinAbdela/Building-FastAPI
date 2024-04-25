from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from utils import load_model, predict_digit, format_image
from PIL import Image
import uvicorn
import sys

app = FastAPI()

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    image = Image.open(file.file)
    data_point = format_image(image)
    model_path = request.app.state.model_path
    model = load_model(model_path)
    digit = predict_digit(model, data_point)
    return JSONResponse(content={"digit": digit}, media_type="application/json")

if __name__ == "__main__":  

    model_path = sys.argv[1]
    app.state.model_path = model_path

    uvicorn.run(app, host="127.0.0.1", port=8080)