import base64
import sys
import matplotlib.pyplot as plt
import numpy as np
from src import send_image
import argparse

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
url = "http://127.0.0.1:8089/predict"

def transform_image(img_path):
    try:
        with open(img_path, "rb") as f:
            im_bytes = f.read()
    except Exception:
        print("Cannot open image: " + img_path)
        exit(1)

    return base64.b64encode(im_bytes).decode("utf8")

def get_prediction(img, url, explain, precision):
    data = send_image(img, url, explain, precision)
    
    if not data.get('success'):
        print("Error: " + data.get('error'))
    else:
        prediction = data.get('prediction')
        probability = data.get('probability')
        explain = data.get('image')
        
        print(f"prediction: {prediction} - {round(probability, 2) * 100}%")
        
        if explain is not None:
            plt.imshow(np.array(explain))
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true", help="If the server should explain the prediction")
    parser.add_argument("--precision", default="Moyenne", help="Precision of the explanation (Faible, Moyenne, Importante)")
    parser.add_argument("image", nargs=1, help='path to the image')
    args = parser.parse_args()

    if len(sys.argv) < 2:
        exit(0)
        
    img = transform_image(args.image[0])
    get_prediction(img, url, args.explain, args.precision)

