import torch
import logging
from datetime import datetime

def setup_logging():
    log_dir = 'training_log'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def generate_anchor_embeddings_batch(sequences, tokenizer, model, device, batch_size=32):
    model.to(device)
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.esm(**inputs).last_hidden_state
            mean_output = output[:, 1:output.size(1)].mean(dim=1)
        embeddings.append(mean_output.cpu())
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

def generate_mol_embeddings_batch(smiles_list, tokenizer, model, device, batch_size=32):
    model.to(device)
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch_smiles, padding=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            mol_embedding = outputs.pooler_output
        embeddings.append(mol_embedding.cpu())
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)