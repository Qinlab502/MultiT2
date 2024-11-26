import os
import logging
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import EsmTokenizer, EsmForMaskedLM, AutoTokenizer, AutoModel
from peft import PeftModel

from datasets import ClassificationDataset, ContrastiveDataset, classification_collate_fn, contrastive_collate_fn
from models import Coembedding, ContrastiveLoss
from evaluate import evaluate_model
from utils import setup_logging, generate_mol_embeddings_batch
from config import CONFIG  # Import configuration


def main():
    # Set up logging
    setup_logging()
    logging.info("Starting training process.")

    # Load configuration
    config = CONFIG
    device = config['device']

    # Load classification data
    classification_data_path = config['classification_data_path']
    if not os.path.exists(classification_data_path):
        logging.error(f"Data file not found: {classification_data_path}")
        return
    classification_df = pd.read_excel(classification_data_path)
    classification_df = classification_df.drop_duplicates(subset=['canonicalsmiles'])
    
    logging.info(f"Successfully loaded classification data, sample size: {len(classification_df)}")

    # Load contrastive learning data
    contrastive_data_path = config['contrastive_data_path']
    if not os.path.exists(contrastive_data_path):
        logging.error(f"Data file not found: {contrastive_data_path}")
        return
    contrastive_df = pd.read_excel(contrastive_data_path)
    contrastive_df = contrastive_df.drop_duplicates(subset=['canonicalsmiles'])
    logging.info(f"Successfully loaded contrastive learning data, sample size: {len(contrastive_df)}")

    # Load protein model and tokenizer
    model_name = config['protein_model_name']
    prot_tokenizer = EsmTokenizer.from_pretrained(model_name)
    base_model = EsmForMaskedLM.from_pretrained(model_name)
    prot_model = PeftModel.from_pretrained(base_model, config['protein_peft_path'])
    logging.info("Successfully loaded protein model and tokenizer.")

    # Load molecule model and tokenizer
    mol_model_path = config['molecule_model_path']
    mol_tokenizer = AutoTokenizer.from_pretrained(mol_model_path, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(mol_model_path, deterministic_eval=True, trust_remote_code=True)
    logging.info("Successfully loaded molecule model and tokenizer.")

    # Initialize model
    protein_embedding_dim = config['protein_embedding_dim']
    molecule_embedding_dim = config['molecule_embedding_dim']
    projection_dim = config['projection_dim']

    model = Coembedding(
        molecule_shape=molecule_embedding_dim,
        protein_shape=protein_embedding_dim,
        latent_dimension=projection_dim,
        latent_activation=config['latent_activation'],
        latent_distance=config['latent_distance'],
        classify=True
    ).to(device)
    logging.info("Successfully initialized co-embedding model.")

    # Create ClassificationDataset and ContrastiveDataset instances
    classification_dataset = ClassificationDataset(
        dataframe=classification_df,
        prot_tokenizer=prot_tokenizer,
        prot_model=prot_model,
        device=device
    )
    
    contrastive_dataset = ContrastiveDataset(
        dataframe=contrastive_df,
        prot_tokenizer=prot_tokenizer,
        prot_model=prot_model,
        mol_tokenizer=mol_tokenizer,
        mol_model=mol_model,
        device=device
    )

    logging.info(f"ClassificationDataset and ContrastiveDataset creation completed")

    # Split classification dataset into training and test sets
    train_indices, test_indices = train_test_split(
        list(range(len(classification_dataset))),
        test_size=0.2,
        random_state=42
    )

    train_classification_dataset = Subset(classification_dataset, train_indices)
    test_classification_dataset = Subset(classification_dataset, test_indices)

    # Create DataLoaders
    batch_size = config['batch_size']

    # DataLoader for classification task
    train_classification_loader = DataLoader(
        train_classification_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: classification_collate_fn(batch, prot_tokenizer, prot_model, device),
        pin_memory=True
    )

    test_classification_loader = DataLoader(
        test_classification_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: classification_collate_fn(batch, prot_tokenizer, prot_model, device),
        pin_memory=True
    )
    
    mol_embs = generate_mol_embeddings_batch(list(classification_dataset.index2label.values()), mol_tokenizer, mol_model, device).to(device)

    # DataLoader for contrastive learning task
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: contrastive_collate_fn(batch, prot_tokenizer, prot_model, mol_tokenizer, mol_model, device),
        pin_memory=True
    )

    logging.info(f"DataLoader creation completed, batch size: {batch_size}")

    # Define optimizers and schedulers for both loss functions
    contrastive_opt = torch.optim.AdamW(model.parameters(), lr=config['lr_con'])
    contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(contrastive_opt, T_0=config['T_0'])

    classfication_opt = torch.optim.AdamW(model.parameters(), lr=config['lr_ce'])
    classfication_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(classfication_opt, T_0=config['T_0'])

    # Define loss functions
    loss_fn = ContrastiveLoss()
    classfication_fn = nn.CrossEntropyLoss()
    
    logging.info("Optimizers, schedulers and loss functions initialization completed.")

    os.makedirs(config['model_weights_dir'], exist_ok=True)

    best_test_acc = 0.0 
    best_epoch = 0 

    for epo in range(config['epochs']):
        model.train()
        total_loss = 0

        # Contrastive learning training
        for batch_idx, batch in enumerate(tqdm(contrastive_loader, total=len(contrastive_loader), desc=f"Contrastive Epoch {epo+1}/{config['epochs']}")):
            anchor = batch['anchorEmb'].to(device)
            positive = batch['positiveEmb'].to(device)
        
            anchor_projection = F.normalize(model.protein_projector(anchor), p=2, dim=1)
            positive_projection = F.normalize(model.molecule_projector(positive), p=2, dim=1)
        
            loss = loss_fn(anchor_projection, positive_projection)

            contrastive_opt.zero_grad()
            loss.backward()
            contrastive_opt.step()

            total_loss += loss.item()

        avg_contrastive_loss = total_loss / len(contrastive_loader)
        contrastive_scheduler.step()

        logging.info(f"Contrastive Epoch {epo+1}/{config['epochs']}, Loss: {avg_contrastive_loss:.4f}")

        # Classification task training
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_classification_loader, total=len(train_classification_loader), desc=f"Classification Epoch {epo+1}/{config['epochs']}")):
            prot_embs = batch['prot_emb'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(mol_embs, prot_embs)    
            loss = classfication_fn(logits, labels)

            classfication_opt.zero_grad()
            loss.backward()
            classfication_opt.step()

            total_loss += loss.item()

        avg_classification_loss = total_loss / len(train_classification_loader)
        classfication_scheduler.step()

        logging.info(f"Classification Epoch {epo+1}/{config['epochs']}, Loss: {avg_classification_loss:.4f}")
        
        # Evaluate the model on the training set
        avg_train_loss, train_acc = evaluate_model(model, mol_embs, train_classification_loader, classfication_fn, device)
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Evaluate the model on the test set
        avg_test_loss, test_acc = evaluate_model(model, mol_embs, test_classification_loader, classfication_fn, device)
        logging.info(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save the model if test accuracy improves
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epo + 1
            
            torch.save({
                'epoch': epo + 1,
                'model_state_dict': model.state_dict(),
                'contrastive_opt_state_dict': contrastive_opt.state_dict(),
                'contrastive_scheduler_state_dict': contrastive_scheduler.state_dict(),
                'classfication_opt_state_dict': classfication_opt.state_dict(),
                'classfication_scheduler_state_dict': classfication_scheduler.state_dict(),
                'loss': avg_test_loss,
                'accuracy': best_test_acc,
            }, os.path.join(config['model_weights_dir'], 'best_model.pth'))

            logging.info(f"New best model saved at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")

    logging.info(f"Training completed. Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()