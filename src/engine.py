""" 
File for training and testing our models.
"""

import torch

from torchmetrics import Accuracy

from tqdm.auto import tqdm
from typing import List, Dict, Tuple

def train_step(model: troch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn: torchmetrics.Accuracy,
               optimizer: torch.optim.Optimizer,
               device: torch.device)->Tuple[float, float]:
    """ 
    Trains a PyTorch model for a single epoch.

    Arguments:
        - model: Pytorch model to be trained.
        - dataloder: DataLoader for the model to be trained on.
        - loss_fn: Pytorch criterion to minimize.
        - acc_fn: Pytorch accuracy metric.
        - optimizer: Optimize to help minimize the loss function.
        - device: Target device to compute on.

    Returns:
        - Tuple saving train training loss and train accuracy metrics.
    """

    # Put model in training mode
    model.train()

    # Initialization of train loss and accuracy
    train_loss, train_acc = 0, 0

    for batch, (X_train, y_train) in enumerate(dataloader):
        # Send data to target device
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass
        y_train_pred = model(X_train)

        # Calculate loss and accuracy
        loss = loss_fn(y_train_pred, y_train)
        train_loss += loss

        accuracy = acc_fn(y_train_pred,y_train)
        train_acc += accuracy

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return (train_loss, train_acc)

def test_step(model: troch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            acc_fn: torchmetrics.Accuracy,
            device: torch.device)->Tuple[float, float]:
    """ 
    Tests a PyTorch model for a single epoch.

    Arguments:
        - model: Pytorch model to be tested.
        - dataloder: DataLoader for the model to be tested on.
        - loss_fn: Pytorch criterion to minimize.
        - acc_fn: Pytorch accuracy metric.
        - device: Target device to compute on.

    Returns:
        - Tuple saving test loss and test accuracy metrics.
    """

    # Put model in evaluation mode
    model.eval()

    # Initialization of test loss and accuracy
    test_loss, test_acc = 0, 0

    # Disables gradient tracking to save memory and speed up computation during testing
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(dataloader):
            # Send data to target device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Forward pass
            y_test_pred = model(X_test)

            # Calculate loss and accuracy
            loss = loss_fn(y_test_pred, y_test)
            test_loss += loss

            accuracy = acc_fn(y_test_pred, y_test)
            test_acc += accuracy

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return (test_loss, test_acc)

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          acc_fn: torchmetrics.Accuracy,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device)->Dict[str, list]:
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Argumentss:
        - model: A PyTorch model to be trained and tested.
        - train_dataloader: DataLoader for the model to be trained on.
        - test_dataloader: DataLoader for the model to be tested on.
        - loss_fn: Pytorch criterion to minimize.
        - acc_fn: Pytorch accuracy metric.
        - epochs: How many epochs to train for.
        - device: Target device to compute on.

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
    """
    # Initialize results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           acc_fn=acc_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        acc_fn=acc_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
