import torch
from torch.utils.data import Dataset
from .utils import generate_anchor_embeddings_batch, generate_mol_embeddings_batch

class ContrastiveDataset(Dataset):
    def __init__(self, dataframe, prot_tokenizer, prot_model, mol_tokenizer, mol_model, device):
        self.data = dataframe
        self.prot_tokenizer = prot_tokenizer
        self.prot_model = prot_model.to(device)
        self.mol_tokenizer = mol_tokenizer
        self.mol_model = mol_model.to(device)
        self.device = device

        self.prot_model.eval()
        self.mol_model.eval()
        for param in self.prot_model.parameters():
            param.requires_grad = False
        for param in self.mol_model.parameters():
            param.requires_grad = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor = row['sequence']
        positive = row['canonicalsmiles']
        return anchor, positive


class ClassificationDataset(Dataset):
    def __init__(self, dataframe, prot_tokenizer, prot_model, device):
        self.data = dataframe
        self.prot_tokenizer = prot_tokenizer
        self.prot_model = prot_model.to(device)
        self.device = device

        self.prot_model.eval()
        for param in self.prot_model.parameters():
            param.requires_grad = False

        self.index2label = {index: canonicalsmiles for index, canonicalsmiles in enumerate(self.data.canonicalsmiles.unique())}
        self.label2index = {canonicalsmiles: index for index, canonicalsmiles in enumerate(self.data.canonicalsmiles.unique())}
        self.data['num_labels'] = self.data.canonicalsmiles.map(self.label2index).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['sequence'], row[['num_labels']]


def contrastive_collate_fn(batch, prot_tokenizer, prot_model, mol_tokenizer, mol_model, device):
    anchors, positives = zip(*batch)
    anchor_embs = generate_anchor_embeddings_batch(anchors, prot_tokenizer, prot_model, device)
    positive_embs = generate_mol_embeddings_batch(positives, mol_tokenizer, mol_model, device)
    return {
        'anchorEmb': anchor_embs,
        'positiveEmb': positive_embs,
    }


def classification_collate_fn(batch, prot_tokenizer, prot_model, device):
    proteins, labels = zip(*batch)
    prot_embs = generate_anchor_embeddings_batch(proteins, prot_tokenizer, prot_model, device)
    return {
        'prot_emb': prot_embs,
        'labels': torch.tensor(labels, dtype=torch.long).squeeze(-1)
    }