""" 
File for Pytorch code of the ViT model to instantiate.
"""

import torch
from torch import nn
from torch import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, datasets

COLOR_CHANNELS = 3
IMG_SIZE = 224
PATCH_SIZE = 16
EMBEDDING_DIM = 768
BATCH_SIZE = 12
NUM_HEADS = 12
MLP_SIZE = 3072
TRANSFORMER_LAYER_NUM = 12
EMBEDDING_DROPOUT = 0.1
CLASSES_NUM=1000
PATCH_NUMBER = ( IMG_SIZE * IMG_SIZE ) // PATCH_SIZE ** 2

class PatchEmbedding(nn.Module):
  """
  Turns a 2D input image into a 1D set of embedded patches.
  """
  def __init__(self,
               in_channels=COLOR_CHANNELS,
               patch_size=PATCH_SIZE,
               embedding_dim=EMBEDDING_DIM):
    """
    Arguments:
      - in_channels = Number of color channel for the input image. Default 3.
      - patch_size = Size of the patches to convert input image into. Default 16.
      - embedding_dim = Size of the embedding vector to turn image into. Default 768.
    """
    super().__init__()

    self.patch_size = patch_size

    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0) # No padding here

    self.flatter = nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x):
    # Prior size verification
    img_res = x.shape[-1]
    assert img_res % self.patch_size == 0, "Image resolution must be divisible by the patch size"

    x_patches = self.patcher(x)
    x_flattened = self.flatter(x_patches)
    x_embedded = x_flattened.permute(0, 2, 1)
    return x_embedded

class MultiHeadAttentionBlock(nn.Module):
  """
  Implements the multi head self attention block of the trasformer encoder.
  """
  def __init__(self,
                embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                attn_dropout:float=0):
    """
    Arguments:
      -embedding_dim: The constant latent vector size D used throughout the Transformer.
      -num_heads: Number of attention heads (k).
      -attn_dropout: Dropout probability applied to the attention weights.
    """
    super().__init__()

    self.normalizer = nn.LayerNorm(normalized_shape=embedding_dim)

    self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 dropout=attn_dropout,
                                                 batch_first=True)

  def forward(self, x):
    x = self.normalizer(x)
    attn_output, _ = self.multihead_attn(query=x,
                                         key=x,
                                         value=x)
    return attn_output

class MLPBlock(nn.Module):
  """
  Implements the MLP block of the transformer encoder.
  """
  def __init__(self,
               embedding_dim=EMBEDDING_DIM,
               mlp_size=MLP_SIZE,
               mlp_dropout:float=0):
    super().__init__()

    self.normalizer = nn.LayerNorm(normalized_shape=embedding_dim)

    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim, out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=mlp_dropout),
        nn.Linear(in_features=mlp_size, out_features=embedding_dim),
        nn.Dropout(p=mlp_dropout))

  def forward(self, x):
    x = self.normalizer(x)
    x = self.mlp(x)
    return x

class TransformerEncoder(nn.Module):
  """
  Create Transformer encoder block.
  """
  def __init__(self,
               embedding_dim=EMBEDDING_DIM,
               num_heads=NUM_HEADS,
               mlp_size=MLP_SIZE,
               attn_dropout:float=0,
               mlp_dropout:float=0):
    super().__init__()

    self.msa_block = MultiHeadAttentionBlock(embedding_dim=embedding_dim,
                                             num_heads=num_heads,
                                             attn_dropout=attn_dropout)

    self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                              mlp_size=mlp_size,
                              mlp_dropout=mlp_dropout)

  def forward(self, x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x
    return x

class ViT(nn.Module):
  """
  Create Vision Transformer architecture model.
  """
  def __init__(self,
               img_size=IMG_SIZE, # Training resolution
               in_channels=COLOR_CHANNELS, # Number of color channels in input image
               patch_size=PATCH_SIZE, # Patch size
               transformer_layer_num=TRANSFORMER_LAYER_NUM, # Number of ViT layers from ViT paper table
               embedding_dim=EMBEDDING_DIM, # Hidden D size from ViT paper table
               mlp_size=MLP_SIZE, # MLP size from ViT paper table
               num_heads=NUM_HEADS, # Number of heads for MSA from ViT paper table
               attn_dropout:float=0, # Dropout for attention from ViT paper table
               mlp_dropout:float=0, # Dropout for MLP layers from ViT paper table
               embedding_dropout=EMBEDDING_DROPOUT, # Dropout for patch and positional embedding
               num_classes=CLASSES_NUM): # Number of classes to predict
    super().__init__()

    # Make sure the image size is divisible by the patch size
    assert img_size % patch_size == 0, "Image resolution must be divisible by the patch size"

    # Number of patches
    self.num_patches = (img_size * img_size) // patch_size ** 2

    # Create learnable class embedding
    self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))

    # Create learnable positional embedding
    self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

    # Dropout value for patch and positional embedding
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    # Create patch embedding layer
    self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    # Create Transformer blocks
    self.transformer_layer = nn.Sequential(*[TransformerEncoder(embedding_dim=embedding_dim,
                                                                num_heads=num_heads,
                                                                mlp_size=mlp_size,
                                                                attn_dropout=attn_dropout,
                                                                mlp_dropout=mlp_dropout) for _ in range(transformer_layer_num)])

    # Create classifier head
    self.classifier = nn.Sequential(
      nn.LayerNorm(normalized_shape=embedding_dim),
      nn.Linear(in_features=embedding_dim, out_features=num_classes))

  def forward(self, x):
    # Get batch size
    batch_size = x.shape[0]

    # Create class token embeddding and expand it to the batch size
    class_token = self.class_embedding.expand(batch_size, -1, -1)

    # Apply patch embedding
    x = self.patch_embedding(x)

    # Concatenate class embedding and patch embedding
    x = torch.cat((class_token, x), dim=1)

    # Add positional embedding
    x = x + self.positional_embedding

    # Apply dropout to embedding part
    x = self.embedding_dropout(x)

    # Pass patch, class and positional embedding through the tranformer blocks
    x = self.transformer_layer(x)

    # 0 logit for classifier
    x = self.classifier(x[:, 0])

    return x

