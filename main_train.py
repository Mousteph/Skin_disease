import argparse

from src import HAM10000, HAM10000_model, Trainer

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
from torch import nn

train_transform = transforms.Compose(
    [
        transforms.GaussianBlur(kernel_size=(5, 5)),
        transforms.RandomPerspective(),
        transforms.RandomRotation(180),
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
    parser.add_argument("--modelname", type=str, default="model/model_mlbio.pth",
                        help="Name of the model to save. Default: model/model_mlbio.pth")
    parser.add_argument("--fine_tune", action="store_true", help="If the model should be fine tuned")
    args = parser.parse_args()
    
    dataset_train = HAM10000.load_from_file(args.root[0], train=True, transform=train_transform)
    dataset_test = HAM10000.load_from_file(args.root[0], train=False, transform=test_transform)
    
    batch_size = 64
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = HAM10000_model(7).to(device, fine_tune=args.fine_tune or False)
    loss_function = nn.CrossEntropyLoss()
    if args.fine_tune:
        optimizer = torch.optim.Adam(model.fc.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters())
        
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    trainer = Trainer(model, optimizer, loss_function, device, scheduler=lr_scheduler)
    trainer.train(train_data, test_data, args.epochs)
    
    trainer.save(args.modelname)