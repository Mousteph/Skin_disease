import requests
import base64
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
url = "http://127.0.0.1:8089/predict"

def transform_image(img_path):
    with open(img_path, "rb") as f:
        im_bytes = f.read()

    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    return im_b64

def send_image(img, url, explain):
    payload = json.dumps({"image": img, "explain": explain})
    response = requests.post(url, data=payload, headers=headers)

    try:
        data = response.json()
        explain = data.get('image')
        
        print(f"prediction: {data.get('prediction')} - {data.get('probability')}")
        
        if explain is not None:
            plt.imshow(np.array(explain))
            plt.show()

    except requests.exceptions.RequestException:
        print(response.text)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true", help="Have an explanation")
    parser.add_argument("image", nargs='+', help='path to the image')
    args = parser.parse_args()

    if len(sys.argv) < 2:
        exit(0)

    img = transform_image(args.image[0])
    send_image(img, url, args.explain)

