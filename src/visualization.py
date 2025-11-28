"""
Visualization module for embeddings and results
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm

# Use modern matplotlib style (seaborn-v0_8 is deprecated)
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback to default style if seaborn-v0_8 is not available
    plt.style.use('default')
sns.set_palette("husl")


def extract_embeddings(model, data_loader, device, n_samples=1000, return_texts=False):
    """Extract encoder embeddings from model"""
    model.eval()
    embeddings = []
    texts = []  # Store source texts
    lengths = []  # Store sentence lengths
    
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            if count >= n_samples:
                break
            
            src = torch.tensor(batch['src'], dtype=torch.long).to(device)
            src_lengths = torch.tensor(batch['src_lengths'], dtype=torch.long).to(device)
            
            # Get encoder output
            if 'Transformer' in model.__class__.__name__:
                src_mask = (src == 0)
                encoder_output = model.encode(src, src_key_padding_mask=src_mask)
                # Use mean pooling: encoder_output is [seq_len, batch_size, d_model]
                # Transpose to [batch_size, seq_len, d_model]
                encoder_output = encoder_output.transpose(0, 1)
                mask = ~src_mask.unsqueeze(-1).float()
                pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                encoder_output, _ = model.encode(src, src_lengths)
                # encoder_output is [batch_size, seq_len, hidden_dim * 2]
                # Use mean pooling
                mask = torch.arange(encoder_output.size(1), device=device).unsqueeze(0) < src_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            embeddings.append(pooled.cpu().numpy())
            
            if return_texts:
                texts.extend(batch.get('src_text', [''] * pooled.size(0)))
                lengths.extend([l.item() for l in src_lengths])
            
            count += pooled.size(0)
            
            if count >= n_samples:
                break
    
    embeddings = np.concatenate(embeddings, axis=0)[:n_samples]
    
    if return_texts:
        return embeddings, texts[:n_samples], lengths[:n_samples]
    return embeddings


def visualize_tsne(embeddings, labels=None, save_path="figures/tsne_visualization.png", 
                   perplexity=30, n_iter=1000):
    """Visualize embeddings using t-SNE"""
    print(f"Running t-SNE on {len(embeddings)} samples...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
    
    plt.title('t-SNE Visualization of Encoder Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {save_path}")
    plt.close()
    
    return embeddings_2d


def visualize_clusters(embeddings, n_clusters=5, save_path="figures/cluster_visualization.png"):
    """Visualize KMeans clustering of embeddings"""
    print(f"Running KMeans clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Use t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'KMeans Clustering (k={n_clusters}) of Encoder Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved cluster visualization to {save_path}")
    plt.close()
    
    return cluster_labels


def plot_training_curves(history, save_path="figures/training_curves.png"):
    """Plot training and validation loss curves"""
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_translation_examples(examples, save_path="figures/translation_examples.txt"):
    """Save translation examples to file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Translation Examples\n")
        f.write("=" * 80 + "\n\n")
        
        for i, ex in enumerate(examples, 1):
            f.write(f"Example {i}:\n")
            f.write(f"Source:     {ex['source']}\n")
            f.write(f"Target:     {ex['target']}\n")
            f.write(f"Prediction: {ex['prediction']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Saved translation examples to {save_path}")

