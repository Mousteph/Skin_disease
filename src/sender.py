import requests
import json
import numpy as np

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def send_image(img: str, url: str, explain: bool) -> tuple:
    """Send an image to the server

    Args:
        img (str): Image encoded in base64
        url (str): URL of the server
        explain (bool): Should the server explain the prediction

    Returns:
        tuple[bool, np.array]: Tuple containing a boolean and the explanation.
        If the boolean is True, the explanation is a tuple containing the prediction, the probability and the explanation.
        Else, the explanation is a string containing the error message.
    """
    
    payload = json.dumps({"image": img, "explain": explain})
    response = requests.post(url, data=payload, headers=headers)

    try:
        data = response.json()
        explain = data.get('image')
        
        prediction = data.get('prediction')
        probability = data.get('probability')
        explain = np.array(explain) if explain is not None else None
        
        return True, (prediction, probability, explain)

    except requests.exceptions.RequestException:
        False, response.text