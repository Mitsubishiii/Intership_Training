"""
Trains a PyTorch Vision Transformer (ViT) model using device-agnostic code.
Can be controlled via command line arguments for hyperparameter tuning.
"""

import os
import torch
import argparse
import torchmetrics
from torchvision import transforms

# Importing local modules from the src/ directory
from src import data_setup, engine, utils, ViT_builder

def main():
    # Setup ArgumentParser
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model on a chosen dataset.")

    # File & Path Arguments
    parser.add_argument("--model_name", 
                        type=str, 
                        default="ViT_Model", 
                        help="The filename for the saved model.")

    parser.add_argument("--dataset", 
                    type=str, 
                    default=None, 
                    help="Name of the dataset")

    parser.add_argument("--train_dir", 
                        type=str, 
                        default="data/pizza_steak_sushi/train", 
                        help="Directory path for training data.")

    parser.add_argument("--test_dir", 
                        type=str, 
                        default="data/pizza_steak_sushi/test", 
                        help="Directory path for testing data.")

    # General Hyperparameters
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=32, 
                        help="Number of images per batch.")

    parser.add_argument("--lr", 
                        type=float, 
                        default=0.001, 
                        help="Learning rate for the optimizer.")

    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=5, 
                        help="Number of training epochs.")

    # ViT Specific Hyperparameters (Optimized for 64x64 by default)
    parser.add_argument("--img_size", 
                        type=int, 
                        default=64, 
                        help="Input image resolution.")

    parser.add_argument("--patch_size", 
                        type=int, 
                        default=8, 
                        help="Patch size (must divide img_size).")

    parser.add_argument("--embedding_dim", 
                        type=int, 
                        default=128, 
                        help="Hidden dimension size D.")

    parser.add_argument("--mlp_size", 
                        type=int, 
                        default=512, 
                        help="MLP hidden size.")

    parser.add_argument("--num_heads", 
                        type=int, 
                        default=8, 
                        help="Number of attention heads.")

    parser.add_argument("--num_layers", 
                        type=int, 
                        default=8, 
                        help="Number of transformer encoder layers.")

    parser.add_argument("--dropout", 
                        type=float, 
                        default=0.1, 
                        help="Dropout rate for attention and MLP.")

    # Parse the arguments
    args = parser.parse_args()

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create transforms (using the img_size argument)
    data_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=data_transform,
        batch_size=args.batch_size,
        dataset_type=args.dataset
    )

    # Initialize ViT model from ViT_builder.py
    model = ViT_builder.ViT(
        img_size=args.img_size,
        in_channels=3,
        patch_size=args.patch_size,
        transformer_layer_num=args.num_layers,
        embedding_dim=args.embedding_dim,
        mlp_size=args.mlp_size,
        num_heads=args.num_heads,
        attn_dropout=args.dropout,
        mlp_dropout=args.dropout,
        embedding_dropout=args.dropout,
        num_classes=len(class_names)
    ).to(device)

    # Set loss, optimizer and accuracy function
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 acc_fn=acc_fn,
                 epochs=args.num_epochs,
                 device=device)

    # Save the model with help from utils.py
    MODEL_NAME = args.model_name if args.model_name.endswith(".pth") else args.model_name + ".pth"
    utils.save_model(model=model,
                     target_dir="models",
                     model_name=MODEL_NAME)
    print(f"Model saved to: models/{MODEL_NAME}")

if __name__ == "__main__":
    main()
