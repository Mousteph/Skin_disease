from flask import Flask, request, abort
from neural_network import MLBioNN
import base64
import json
import io
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

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

model = MLBioNN(len(LESION_TYPE))
model.load_state_dict(torch.load("model_mlbio_cpu.pth"))

transform = transforms.Compose([
            transforms.ToTensor(), # Scale image to [0, 1]
        ])

def do_prediction(img):
    img = transform(img)

    with torch.no_grad():
        pred = model(img.unsqueeze(0))
        pred = pred.argmax(1).item()

    return LESION_TYPE.get(pred)

@app.route("/predict", methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        abort(400)

    img = request.json['image']
    img_bytes = base64.b64decode(img.encode('utf-8'))
    img = Image.open(io.BytesIO(img_bytes))

    return json.dumps({"prediction": do_prediction(img)})


if __name__ == "__main__":
    app.run("0.0.0.0", port=8089)
