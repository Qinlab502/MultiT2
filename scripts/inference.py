import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.tree import DecisionTreeClassifier
from model import Coembedding
from utils import generate_anchor_embeddings_batch, generate_mol_embeddings_batch
from transformers import EsmTokenizer, EsmForMaskedLM, AutoTokenizer, AutoModel
from peft import PeftModel


def load_models(device):
    """
    Load protein and molecule models along with their tokenizers.
    """
    # Load protein model and tokenizer
    model_name = '../models/esm2/esm2_t33_650M_UR50D'
    prot_tokenizer = EsmTokenizer.from_pretrained(model_name)
    base_model = EsmForMaskedLM.from_pretrained(model_name)
    prot_model = PeftModel.from_pretrained(base_model, '../models/plm')

    # Load molecule model and tokenizer
    mol_model_path = "../models/ibm/MoLFormer-XL-both-10pct"
    mol_tokenizer = AutoTokenizer.from_pretrained(mol_model_path, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(mol_model_path, deterministic_eval=True, trust_remote_code=True)

    prot_model.to(device)
    mol_model.to(device)

    return prot_tokenizer, prot_model, mol_tokenizer, mol_model


def generate_embeddings(data_file, prot_tokenizer, prot_model, mol_tokenizer, mol_model, device):
    """
    Generate embeddings for protein sequences and molecule SMILES.
    """
    data = pd.read_excel(data_file)
    data = data.drop_duplicates(subset=['canonicalsmiles'])

    prot_seq = data['sequence'].tolist()
    mol_smiles = data['canonicalsmiles'].tolist()

    print("Generating protein embeddings...")
    prot_emb = generate_anchor_embeddings_batch(prot_seq, prot_tokenizer, prot_model, device)

    print("Generating molecule embeddings...")
    mol_emb = generate_mol_embeddings_batch(mol_smiles, mol_tokenizer, mol_model, device)

    return prot_emb, mol_emb, data


def run_inference(model, prot_emb, mol_emb, device):
    """
    Perform inference using the Coembedding model.
    """
    print("Running inference...")
    with torch.no_grad():
        prot_emb = prot_emb.to(device)
        mol_emb = mol_emb.to(device)

        # Compute similarity matrix
        similarity_matrix = model(mol_emb, prot_emb)

        # Apply softmax normalization
        similarity_matrix_softmax = F.softmax(similarity_matrix, dim=1)

        # Find the maximum similarity values for each molecule
        max_similarities = torch.max(similarity_matrix_softmax, dim=1).values

    return similarity_matrix_softmax, max_similarities


def train_classifier(positive_samples, negative_samples):
    """
    Train a Decision Tree Classifier using the similarity scores.
    """
    print("Training Decision Tree Classifier...")
    X = np.concatenate([positive_samples, negative_samples])
    y = np.concatenate([np.ones(len(positive_samples)), np.zeros(len(negative_samples))])

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X.reshape(-1, 1), y)  # Reshape to sklearn's input format

    return clf


def predict_test_set(clf, max_similarities):
    """
    Predict the test set using the trained classifier.
    """
    print("Predicting test set...")
    y_pred = clf.predict(max_similarities.reshape(-1, 1))
    return y_pred


def get_top_molecules(similarity_matrix_softmax, mol_smiles, data, top_k=5):
    """
    Extract the top K similar molecules for each protein.
    """
    print(f"Extracting top {top_k} similar molecules...")
    top_indices = torch.argsort(-similarity_matrix_softmax, dim=1)[:, :top_k]
    top_similarities = torch.gather(-similarity_matrix_softmax, 1, top_indices)

    name_to_smiles = dict(zip(data['canonicalsmiles'], data['T2PKproductsname']))

    results = []
    for i, (indices, similarities) in enumerate(zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy())):
        protein_results = []
        for index, similarity in zip(indices, similarities):
            smiles = mol_smiles[index]
            molecule_label = name_to_smiles.get(smiles, "Unknown")
            protein_results.append({
                "Protein_Index": i,
                "Molecule_Label": molecule_label,
                "SMILES": smiles,
                "Similarity": -similarity
            })
        results.append(protein_results)

    return results


def main(fasta_file, model_weight_path, output_file, top_k=5, input_file='../data/T2_data_norm.csv'):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models and tokenizers
    prot_tokenizer, prot_model, mol_tokenizer, mol_model = load_models(device)

    # Generate embeddings
    prot_emb, mol_emb, data = generate_embeddings(input_file, prot_tokenizer, prot_model, mol_tokenizer, mol_model, device)

    # Load Coembedding model
    model = Coembedding(
        molecule_shape=768,
        protein_shape=1280,
        latent_dimension=1024,
        latent_activation=torch.nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        temperature=0.1
    ).to(device)

    checkpoint = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Inference
    similarity_matrix_softmax, max_similarities = run_inference(model, prot_emb, mol_emb, device)

    # Extract diagonal and non-diagonal elements for training
    output_np = similarity_matrix_softmax.cpu().detach().numpy()
    positive_samples = output_np.diagonal()
    negative_samples = np.mean(output_np - np.diag(np.diag(output_np)), axis=0)

    # Train Decision Tree Classifier
    clf = train_classifier(positive_samples, negative_samples)

    # Read protein sequences from FASTA file
    prot_seq = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

    # Generate protein embeddings
    prot_emb = generate_anchor_embeddings_batch(prot_seq, prot_tokenizer, prot_model, device)

    # Inference on test set
    similarity_matrix_softmax, max_similarities = run_inference(model, prot_emb, mol_emb, device)

    # Predict test set
    y_pred = predict_test_set(clf, max_similarities.cpu().detach().numpy())

    # Extract top K similar molecules for each protein
    top_molecules = get_top_molecules(similarity_matrix_softmax, data['canonicalsmiles'].tolist(), data, top_k=top_k)

    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        for i, (protein_results, pred) in enumerate(zip(top_molecules, y_pred)):
            if pred == 0:
                f.write(f"Protein {i}: Unknown product\n\n")
            else:
                f.write(f"Protein {i} top {top_k} similar molecules:\n")
                for result in protein_results:
                    f.write(f"  - Molecule Label: {result['Molecule_Label']}, "
                            f"SMILES: {result['SMILES']}, "
                            f"Similarity: {result['Similarity']:.4f}\n")
                f.write("\n")

    print("Inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on protein and molecule embeddings.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--model_weight_path", type=str, required=True, help="Path to the model weight file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output results.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top similar molecules to output per protein.")
    args = parser.parse_args()

    main(args.fasta_file, args.model_weight_path, args.output_file, args.top_k)