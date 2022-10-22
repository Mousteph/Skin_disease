import requests
import base64
import json
import sys

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
url = "http://127.0.0.1:8089/predict"

def transform_image(img_path):
    with open(img_path, "rb") as f:
        im_bytes = f.read()

    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    return im_b64

def send_image(img, url):
    payload = json.dumps({"image": img})
    response = requests.post(url, data=payload, headers=headers)

    try:
        data = response.json()
        print(data)
    except requests.exceptions.RequestException:
        print(response.text)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit(0)

    img = transform_image(sys.argv[1])
    send_image(img, url)

