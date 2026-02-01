"""
Trains a PyTorch model using device-agnostic code.
Can be controlled via command line arguments for hyperparameter tuning.
"""

import os
import torch
import argparse
import torchmetrics
from torchvision import transforms

# Importing local modules
from src import data_setup, engine, utils, CNN_builder

def main():
    # Setup ArgumentParser
    parser = argparse.ArgumentParser(description="Train a Pytorch model on a choosen dataset.")

    # Add Arguments
    parser.add_argument("--model_name", 
                        type=str, 
                        default="CNN", 
                        help="The filename for the saved model (e.g., 'CNN').")

    parser.add_argument("--train_dir",
                        type=str)

    parser.add_argument("--test_dir", 
                        type=str)

    parser.add_argument("--dataset", 
                    type=str, 
                    default=None, 
                    help="Name of the dataset")

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=32, 
                        help="Number of images per batch (default: 32).")

    parser.add_argument("--lr", 
                        type=float, 
                        default=0.001, 
                        help="Learning rate for the optimizer (default: 0.001).")

    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=5, 
                        help="Number of training epochs (default: 5).")

    parser.add_argument("--hidden_units", 
                        type=int, 
                        default=10, 
                        help="Number of hidden units in the neural network (default: 10).")

    args = parser.parse_args()

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=data_transform,
        batch_size=args.batch_size,
        dataset_type=args.dataset
    )

    # Initialize the model
    model = CNN_builder.CNN(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(device)

    # Construct the path to the model file
    model_save_path = os.path.join("models", args.model_name if args.model_name.endswith(".pth") else args.model_name + ".pth")

    if os.path.exists(model_save_path):
        print(f"Loading weights from: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print(f"{model_save_path} not found. Training from scratch.")

    # Set loss, optimizer and accuracy function
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

    # Start training
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           acc_fn=acc_fn,
                           epochs=args.num_epochs,
                           device=device)

    # Save the updated model
    utils.save_model(model=model,
                     target_dir="models",
                     model_name=f"trained_{args.model_name}.pth")

if __name__ == "__main__":
    main()
