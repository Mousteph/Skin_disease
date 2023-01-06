import requests
import json

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def send_image(img: str, url: str, explain: bool, precision: str = "Moyenne") -> dict:
    """Send an image to the server

    Args:
        img (str): Image encoded in base64
        url (str): URL of the server
        explain (bool): Should the server explain the prediction
        precision (str, optional): Precision of the explanation. Defaults to "Moyenne".
            Can be "Faible", "Moyenne" or "Forte".

    Returns:
        dict: Dictionary containing the prediction, the probability and the explanation.
    """
    
    payload = json.dumps({"image": img, "explain": explain, "precision": precision})
    try:
        response = requests.post(url, data=payload, headers=headers)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}
