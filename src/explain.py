import torch
import torch.nn.functional as F

import numpy as np

from lime import lime_image
from skimage.segmentation import mark_boundaries


class ExplainResults:
    def __init__(self, torch_model, transform, lesion_type):
        self.lesion_type = lesion_type
        self.torch_model = torch_model
        self.transform = transform
        
        self.explainer = lime_image.LimeImageExplainer()
        
    def batch_prediction(self, images):
        batch = torch.stack(tuple(self.transform(i) for i in images), dim=0)
        self.torch_model.eval()
        
        with torch.no_grad():
            pred = self.torch_model(batch)
            proba = F.softmax(pred, dim=1)

            return proba.numpy()
        
    def prediction(self, image, explain=False, num_samples=10):
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
                                                        positive_only=True,
                                                        num_features=5,
                                                        hide_rest=False)

            return lesion, mark_boundaries(temp/255.0, mask)
        
        return lesion, None
