import torchvision.models as models
from torch import nn

def HAM10000_model(output_dim: int, fine_tune=False) -> nn.Module:
    """Return a model for the HAM10000 dataset.

    Args:
        output_dim (int): Dimension of the output.
        fine_tune (bool, optional): If True, fine tune the model (train the last layer). Defaults to False.

    Returns:
        nn.Module: Model for the HAM10000 dataset.
    """
    
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    
    if fine_tune:
        for param in model.parameters():
            param.requiresGrad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_dim)
    
    return model