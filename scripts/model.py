import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def __call__(self, anchor_projection, positive_projection):
        return self.loss_fn(anchor_projection, positive_projection)

    def loss_fn(self, anchor_projection, positive_projection):
        similarity_matrix = torch.matmul(anchor_projection, positive_projection.T)
        similarity_matrix = similarity_matrix / self.temperature

        pos_sim = torch.diag(similarity_matrix)

        lprobs_pocket = F.log_softmax(similarity_matrix, dim=1)
        indices = torch.arange(len(pos_sim))
        L_pocket = -lprobs_pocket[indices, indices].mean()

        lprobs_mol = F.log_softmax(similarity_matrix.T, dim=1)
        L_mol = -lprobs_mol[indices, indices].mean()

        loss = 0.5 * (L_pocket + L_mol)
        return loss


class Coembedding(nn.Module):
    def __init__(
        self,
        molecule_shape=768,
        protein_shape=1280,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        temperature=0.1
    ):
        super(Coembedding, self).__init__()
        self.molecule_projector = nn.Sequential(
            nn.Linear(molecule_shape, latent_dimension),
            latent_activation(),
            nn.Linear(latent_dimension, latent_dimension)
        )
        self.protein_projector = nn.Sequential(
            nn.Linear(protein_shape, latent_dimension),
            latent_activation(),
            nn.Linear(latent_dimension, latent_dimension)
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.do_classify = classify

        if classify:
            self.distance_metric = latent_distance
            self.activator = nn.CosineSimilarity(dim=-1)

    def forward(self, molecule, protein):
        if self.do_classify:
            return self.classify(molecule, protein)

    def classify(self, molecule, protein):
        molecule_projection = self.molecule_projector(molecule)
        protein_projection = self.protein_projector(protein)

        molecule_projection = molecule_projection.unsqueeze(0) 
        protein_projection = protein_projection.unsqueeze(1)
        
        distance = self.activator(molecule_projection, protein_projection)
        scaled_distance = distance / self.temperature
        return scaled_distance