import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def evaluate_model(model, mol_embs, test_loader, loss_fn, device):
    """
    Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        mol_embs (torch.Tensor): Precomputed molecule embeddings.
        test_loader (DataLoader): DataLoader for the test set.
        loss_fn (callable): Loss function to compute the test loss.
        device (torch.device): Device to perform evaluation on (CPU or GPU).

    Returns:
        tuple: A tuple containing average test loss and accuracy.
    """
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, batch in enumerate(test_loader):
            prot_embs = batch['prot_emb'].to(device)  # Protein embeddings
            labels = batch['labels'].to(device)      # Ground truth labels

            # Forward pass
            logits = model(mol_embs, prot_embs)
            loss = loss_fn(logits, labels)  # Compute loss
            test_loss += loss.item()

            # Get predictions and save
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)  # Predicted labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_test_loss, accuracy