import torch
import torch.nn as nn

CONFIG = {
    # General training parameters
    'epochs': 100,
    'batch_size': 32,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Data paths
    'classification_data_path': "../data/T2_data_norm.csv",
    'contrastive_data_path': "../data/T2_data_norm.csv",

    # Model parameters
    'protein_model_name': '../models/esm2/esm2_t33_650M_UR50D',
    'protein_peft_path': '../models/plm',
    'molecule_model_path': '../models/ibm/MoLFormer-XL-both-10pct',
    'protein_embedding_dim': 1280,
    'molecule_embedding_dim': 768,
    'projection_dim': 1024,
    'latent_distance': 'Cosine',

    # Learning rates and optimizers
    'lr_con': 1e-4,  # Learning rate for contrastive loss
    'lr_ce': 1e-4,       # Learning rate for classification
    'T_0': 10,           # CosineAnnealingWarmRestarts parameter

    # Loss functions
    'latent_activation': nn.ReLU,

    # Output directories
    'model_weights_dir': '../models',
}