import argparse

from datasetHAM10000 import HAM10000
from model import HAM10000_model
from trainer import Trainer

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn

train_transform = transforms.Compose(
    [
        transforms.RandomPerspective(),
        transforms.RandomRotation(180),
        transforms.GaussianBlur(kernel_size=(5, 5)),
        transforms.ToTensor(), # Scale image to [0, 1]
    ])

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),   
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs=1, help='Root to the images')
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs to train the model. Default: 15")
    parser.add_argument("--modelname", type=str, default="model_mlbio.pth",
                        help="Name of the model to save. Default: model_mlbio.pth")
    args = parser.parse_args()
   
    dataset_train = HAM10000.load_from_file(args.root[0], train=True, transform=train_transform)
    dataset_test = HAM10000.load_from_file(args.root[0], train=False, transform=test_transform)
    
    batch_size = 64
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = HAM10000_model(7).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = Trainer(model, optimizer, loss_function, device)
    trainer.training_process(train_data, test_data, args.epochs)
    
    trainer.save(args.modelname)