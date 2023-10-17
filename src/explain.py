import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple

import numpy as np

from lime import lime_image
from skimage.segmentation import mark_boundaries


class ExplainResults:
    def __init__(self, torch_model: torch.nn.Module, transform: transforms.Compose,
                 lesion_type: dict):
        """Class to explain the results of a model

        Args:
            torch_model (torch.nn.Module): PyTorch model to use
            transform (transforms.Compose): Transformation to apply
            lesion_type (dict): Type of skin diseases
        """
        
        self.lesion_type = lesion_type
        self.torch_model = torch_model
        self.transform = transform
        
        self.explainer = lime_image.LimeImageExplainer()
        
    def batch_prediction(self, images: list) -> np.ndarray:
        """Make a prediction on a batch of images

        Args:
            images (list): List of numpy array

        Returns:
            np.ndarray: The probabilities of the prediction of each images
        """
        
        batch = torch.stack(tuple(self.transform(i) for i in images), dim=0)
        self.torch_model.eval()
        
        with torch.no_grad():
            pred = self.torch_model(batch)
            proba = F.softmax(pred, dim=1)

            return proba.numpy()
        
    def prediction(self, image: np.array, explain: bool = False,
                   num_samples: int = 100) -> Tuple[tuple, np.array]:
        """Make a prediction on a single image

        Args:
            image (np.array): Image to do the prediction
            explain (bool, optional): If an explication is needed. Defaults to False.
            num_samples (int, optional): Number of samples needed to make the explanation. The more the sample the longer the explanation will take. Defaults to 100.

        Returns:
            Tuple[tuple, np.array]: (The prediction, The probability of the prediction), (The explication if explain is True, None if explain is False).
        """

        probs = self.batch_prediction([image])
        val = probs.argmax()
        lesion = (self.lesion_type.get(val), probs[0][val])
        
        if explain:
            explanation = self.explainer.explain_instance(np.array(image), 
                                                          self.batch_prediction, # classification function
                                                          top_labels=1,
                                                          hide_color=0,
                                                          num_samples=num_samples) # number of images that will be sent to classification function

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=False,
                                                        num_features=5,
                                                        hide_rest=False)

            return lesion, mark_boundaries(temp/255.0, mask)
        
        return lesion, None
