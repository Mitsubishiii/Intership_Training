# Internship Training: Deep Learning Vision Project
**Author: CHERIFI Sacha**

## Project Structure

### Notebooks (Research & Development)
The project consists of two distinct development phases:
1. **Exploratory Phase**: Initial notebooks where CNN and ViT models were built in a "monolithic" fashion (all code in a single file). This allowed for model validation without the complexity of modularity.
2. **Modularity Phase (notebooks/modularity.ipynb)**: This notebook acts as the "orchestrator." It contains %%writefile cells used to automatically generate and update the files in the src/ directory and the main training scripts.

### Modular Architecture (src/)
The src/ folder contains the reusable components of the project:
* **data_setup.py**: Prepares DataLoaders. It handles both local image folders and automatic downloads of torchvision datasets like CIFAR-10.
* **CNN_builder.py**: Defines the TinyVGG architecture for the CNN model.
* **ViT_builder.py**: Defines the full Vision Transformer architecture (including PatchEmbedding, MSA, MLP, and TransformerEncoder).
* **.py**: Defines the full Vision Transformer architecture (including PatchEmbedding, MSA, MLP, and TransformerEncoder).
* **engine.py**: Contains universal training and testing loop functions (train_step, test_step, train).
* **utils.py**: Utility functions for saving and loading model weight

### Data and Models
* **data/**: Contains datasets.

* **models/**: Stores saved .pth files after training.

---
### Testing Models

**CNN**

python train_CNN.py --dataset cifar10 --model_name CNN_CIFAR --batch_size 32 --hidden_units 10 --num_epochs 4

**Vision Transformer (ViT)**

python train_ViT.py --dataset cifar10 --model_name ViT_CIFAR10 --num_epochs 4 --batch_size 12 --img_size 32 --patch_size 16 --embedding_dim 768 --mlp_size 3072 --num_heads 12 --num_layers 12 --dropout 0.1

---
