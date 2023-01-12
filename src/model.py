import torchvision.models as models
from torch import nn

def HAM10000_model(output_dim: int) -> nn.Module:
    """Return a model for the HAM10000 dataset.

    Args:
        output_dim (int): Dimension of the output.

    Returns:
        nn.Module: Model for the HAM10000 dataset.
    """
    
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_dim)
    
    return model