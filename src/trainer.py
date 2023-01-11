import torch
from typing import Tuple
import tqdm
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module,
                 device: str) -> None:
        """Initialize the trainer.
        Args:
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            loss (torch.nn.Module): Loss function to use.
            device (str): Device to use (cuda or cpu).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def _train(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Train the model for one epoch.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        """
        
        self.model.train()
        
        for _, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss(pred, y)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _test(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Compute the loss and the accurary for a given dataloader.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        """
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        test_loss /= num_batches
        correct /= size
        
        return test_loss, correct
    
    def save(self, path: str) -> None:
        """Save the model.
        Args:
            path (str): Path to save the model.
        """
        
        torch.save(self.model.to('cpu').state_dict(), path)
        print(f"Model saved to {path}.")
   
    def training_process(self,
                         train_loader: torch.utils.data.DataLoader,
                         test_loader: torch.utils.data.DataLoader,
                         epochs: int = 10,
                         plot_tdqm=False) -> Tuple[list, list, list, list]:
        """Train the model for a given number of epochs.
        
        Args:
            train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
            test_loader (torch.utils.data.DataLoader): Dataloader for the test set.
            epochs (int, optional): Number of epochs. Defaults to 10.
            plot_tdqm (bool, optional): Plot the training process with tqdm. Defaults to False.
            
        Returns:
            Tuple[list, list, list, list]: Train loss, train accuracy, test loss, test accuracy.
        """
        
        train_losses = [0] * epochs
        train_accuracies = [0] * epochs
        test_losses = [0] * epochs
        test_accuracies = [0] * epochs
        
        def one_epoch(train_loader: torch.utils.data.DataLoader,
                      test_loader: torch.utils.data.DataLoader):
            self._train(train_loader)
            train_loss, train_accuracy = self._test(train_loader)
            test_loss, test_accuracy = self._test(test_loader)
            
            return train_loss, train_accuracy, test_loss, test_accuracy
        
        if plot_tdqm:
            with tqdm.tqdm(range(epochs), desc="Epoch") as t:
                for e in t:
                    train_loss, train_accuracy, test_loss, test_accuracy = one_epoch(train_loader, test_loader)
                    
                    train_losses[e] = train_loss
                    train_accuracies[e] = train_accuracy
                    test_losses[e] = test_loss
                    test_accuracies[e] = test_accuracy
                    
                    t.set_postfix(train_loss=train_loss,
                                train_accuracy=train_accuracy,
                                test_loss=test_loss,
                                test_accuracy=test_accuracy)
                    
        else:
            for e in range(epochs):
                train_loss, train_accuracy, test_loss, test_accuracy = one_epoch(train_loader, test_loader)
                
                train_losses[e] = train_loss
                train_accuracies[e] = train_accuracy
                test_losses[e] = test_loss
                test_accuracies[e] = test_accuracy
                
                print(f"Epoch {e + 1}/{epochs}, Train loss: {train_loss:.4f}, "
                      f"Train accuracy: {train_accuracy:.4f}, Test loss: {test_loss:.4f}, "
                      f"Test accuracy: {test_accuracy:.4f}")

        return train_losses, train_accuracies, test_losses, test_accuracies