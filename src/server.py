from flask import Flask, request, abort
from neural_network import MLBioNN
from explain import ExplainResults
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
model.load_state_dict(torch.load("/code/model/model_mlbio_cpu.pth"))

transform = transforms.Compose([
            transforms.ToTensor(), # Scale image to [0, 1]
        ])

explain_model = ExplainResults(model, transform, LESION_TYPE)

@app.route("/predict", methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        abort(400)

    img = request.json['image']
    img_bytes = base64.b64decode(img.encode('utf-8'))
    img = Image.open(io.BytesIO(img_bytes))
    
    should_explain = request.json['explain']
    lesion, mask = explain_model.prediction(img, should_explain)

    if mask is not None:
        mask =  mask.tolist() 

    return json.dumps({"prediction": lesion[0],
                       "probability": float(lesion[1]),
                       "image": mask})


if __name__ == "__main__":
    app.run("0.0.0.0", port=8089)
