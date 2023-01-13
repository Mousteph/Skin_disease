import torchvision.models as models
from torch import nn

def HAM10000_model(output_dim: int, fine_tune=False, model_type='resnet18') -> nn.Module:
    """Return a model for the HAM10000 dataset.

    Args:
        output_dim (int): Dimension of the output.
        fine_tune (bool, optional): If True, fine tune the model (train the last layer). Defaults to False.
        model_type (str, optional): Model to use ('resnet18' or 'resnet34'). Defaults to 'resnet18'.

    Returns:
        nn.Module: Model for the HAM10000 dataset.
    """
    
    if model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    
    if fine_tune:
        for param in model.parameters():
            param.requiresGrad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_dim)
    
    return model