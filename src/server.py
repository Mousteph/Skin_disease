from flask import Flask, request
from model import HAM10000_model
from explain import ExplainResults
import base64
import json
import io
import torch
from torchvision import transforms
from PIL import Image
import os

MODEL = os.environ.get("MODEL", "resnet34")

app = Flask(__name__)

LESION_TYPE = {
    0: 'Melanocytic nevi',
    1: 'dermatofibroma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

PRECISION = {
    "Faible": 50,
    "Moyenne": 200,
    "Importante": 1000,
}

model = HAM10000_model(len(LESION_TYPE), model_type=MODEL)
model.load_state_dict(torch.load(f"/code/model/model_{MODEL}.pth"))

transform = transforms.Compose([
            transforms.ToTensor(), # Scale image to [0, 1]
            transforms.Resize((450, 600)),
        ])

explain_model = ExplainResults(model, transform, LESION_TYPE)

def return_error(error: str):
    return json.dumps({
        "success": False,
        "error": error
    })

@app.route("/predict", methods=['POST'])
def prediction():
    if not request.json or 'image' not in request.json:
        return return_error("Missing image")
        
    try:
        img = request.json['image']
        img_bytes = base64.b64decode(img.encode('utf-8'))
        img = Image.open(io.BytesIO(img_bytes))
        
    except Exception as e:
        return return_error("Invalid image: " + str(e))

    should_explain = request.json['explain'] if 'explain' in request.json else False
    if type(should_explain) is not bool:
        return return_error("Invalid type for explain: Should be a boolean")
        
    precision = request.json['precision'] if 'precision' in request.json else "Moyenne"
    precision = PRECISION.get(precision, 10)
    
    try:
        lesion, mask = explain_model.prediction(img, should_explain, precision)
        mask = mask.tolist() if mask is not None else None
    except Exception as e:
        return return_error("Invalid image: " + str(e))
        
    return json.dumps({
        "success": True,
        "prediction": lesion[0],
        "probability": float(lesion[1]),
        "image": mask
    })    
    
if __name__ == "__main__":
    app.run("0.0.0.0", port=8089)
