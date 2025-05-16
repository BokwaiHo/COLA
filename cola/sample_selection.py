# cola/sample_selection.py
"""
Sample selection module for the COLA framework (Stage 3).
Selects samples to maximize representativeness and diversity in activation space.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger("COLA")

class ActivationHook:
    """Hook for extracting activations from model layers."""
    
    def __init__(self, module):
        """
        Initialize activation hook for a module.
        
        Args:
            module: PyTorch module to hook
        """
        self.activations = None
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Store the output activations of the module."""
        self.activations = output.detach()
    
    def remove(self):
        """Remove the hook."""
        self.hook.remove()

def extract_activations(
    model,
    inputs,
    layers=None,
    batch_size=4,
    device="cuda"
):
    """
    Extract activations from specified layers of the model.
    
    Args:
        model: The model to extract activations from
        inputs: The input tensors (input_ids, attention_mask)
        layers: List of layer indices to extract activations from (None for all)
        batch_size: Batch size for forward pass
        device: Device to run the model on
        
    Returns:
        Dictionary mapping layer names to activations
    """
    # Move to CPU to avoid CUDA OOM
    model = model.to(device)
    model.eval()
    
    # Determine which layers to hook
    if layers is None:
        # Try to automatically detect transformer layers
        transformer_layers = []
        for name, module in model.named_modules():
            if any(layer_type in name for layer_type in ["encoder.layer", "decoder.layer", "layers"]):
                if hasattr(module, "output") or hasattr(module, "feed_forward") or hasattr(module, "ffn"):
                    transformer_layers.append((name, module))
        
        if not transformer_layers:
            logger.warning("Could not automatically detect transformer layers")
            return {}
    else:
        # Use specified layers
        transformer_layers = []
        for i, layer_idx in enumerate(layers):
            layer_name = f"layer.{layer_idx}"
            layer_module = None
            
            # Find module with matching name
            for name, module in model.named_modules():
                if layer_name in name and hasattr(module, "output"):
                    layer_module = module
                    transformer_layers.append((name, module))
                    break
    
    # Register hooks
    hooks = []
    for name, module in transformer_layers:
        # If the module has a feed_forward or ffn attribute, use that
        if hasattr(module, "feed_forward"):
            target_module = module.feed_forward
        elif hasattr(module, "ffn"):
            target_module = module.ffn
        elif hasattr(module, "output"):
            target_module = module.output
        else:
            target_module = module
        
        hook = ActivationHook(target_module)
        hooks.append((name, hook))
    
    # Prepare batches
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    num_samples = input_ids.shape[0]
    all_activations = {}
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)
            
            # Forward pass
            _ = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            # Collect activations from hooks
            for name, hook in hooks:
                activations = hook.activations
                
                # Average over sequence length and batch
                # This gives a single representation per sample and layer
                mean_activations = activations.mean(dim=1)  # Average over sequence length
                
                if name not in all_activations:
                    all_activations[name] = []
                
                all_activations[name].append(mean_activations.cpu())
    
    # Concatenate activations for each layer
    for name in all_activations:
        all_activations[name] = torch.cat(all_activations[name], dim=0)
    
    # Clean up hooks
    for _, hook in hooks:
        hook.remove()
    
    return all_activations

def random_projection(activation_vectors, reduced_dim=64):
    """
    Apply random projection to reduce dimensionality of activation vectors.
    
    Args:
        activation_vectors: Tensor of activation vectors
        reduced_dim: Target dimensionality
        
    Returns:
        Reduced dimensionality vectors
    """
    original_dim = activation_vectors.shape[1]
    
    # Create random projection matrix
    # Following the recommendation from the paper, using normal distribution
    random_matrix = torch.randn(original_dim, reduced_dim) / np.sqrt(reduced_dim)
    
    # Apply projection
    projected_vectors = torch.matmul(activation_vectors, random_matrix)
    
    return projected_vectors

def aggregate_layer_activations(layer_activations):
    """
    Aggregate activations from different layers.
    
    Args:
        layer_activations: Dictionary mapping layer names to activations
        
    Returns:
        Tensor of aggregated activations
    """
    # Stack activations from all layers
    all_layers = []
    for layer_name, activations in layer_activations.items():
        # Normalize each layer's activations
        norm_activations = activations / (activations.norm(dim=1, keepdim=True) + 1e-8)
        all_layers.append(norm_activations)
    
    # Concatenate along feature dimension
    aggregated = torch.cat(all_layers, dim=1)
    
    return aggregated

def cluster_samples(activation_vectors, n_clusters=128, random_state=42):
    """
    Cluster samples based on their activation vectors.
    
    Args:
        activation_vectors: Tensor of activation vectors
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        KMeans object with cluster assignments
    """
    # Convert to numpy for sklearn
    if isinstance(activation_vectors, torch.Tensor):
        vectors_np = activation_vectors.numpy()
    else:
        vectors_np = activation_vectors
    
    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state,
        n_init=10
    )
    kmeans.fit(vectors_np)
    
    return kmeans

def select_representative_samples(samples, activation_vectors, kmeans):
    """
    Select representative samples from each cluster.
    
    Args:
        samples: List of processed samples
        activation_vectors: Tensor of activation vectors
        kmeans: KMeans object with cluster assignments
        
    Returns:
        List of selected representative samples
    """
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # Convert to numpy for distance calculations
    if isinstance(activation_vectors, torch.Tensor):
        vectors_np = activation_vectors.numpy()
    else:
        vectors_np = activation_vectors
    
    # Find the closest sample to each cluster center
    selected_indices = []
    
    for cluster_idx in range(len(cluster_centers)):
        # Get indices of samples in this cluster
        cluster_sample_indices = np.where(cluster_labels == cluster_idx)[0]
        
        if len(cluster_sample_indices) == 0:
            continue
        
        # Get activation vectors for samples in this cluster
        cluster_vectors = vectors_np[cluster_sample_indices]
        
        # Calculate distances to cluster center
        center = cluster_centers[cluster_idx]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        
        # Find the closest sample
        closest_idx_in_cluster = np.argmin(distances)
        closest_idx_overall = cluster_sample_indices[closest_idx_in_cluster]
        
        selected_indices.append(closest_idx_overall)
    
    # Get the selected samples
    selected_samples = [samples[i] for i in selected_indices]
    
    return selected_samples

def select_samples(
    processed_samples: List[Dict],
    model,
    tokenizer,
    device: str = "cuda",
    num_clusters: int = 128,
    reduced_dim: int = 64,
    activation_layers: Optional[List[int]] = None,
    batch_size: int = 4,
    random_state: int = 42
) -> List[Dict]:
    """
    Select diverse and representative samples based on their activation patterns.
    
    Args:
        processed_samples: List of processed samples from Stage 2
        model: The LLM model
        tokenizer: The tokenizer
        device: Device to run the model on
        num_clusters: Number of clusters (determines final sample count)
        reduced_dim: Dimension after random projection
        activation_layers: List of layer indices to use (None for all)
        batch_size: Batch size for activation extraction
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected samples
    """
    logger.info("Starting sample selection based on activation patterns")
    logger.info(f"Using {num_clusters} clusters and {reduced_dim} dimensions after projection")
    
    # Prepare inputs for all samples
    sample_texts = [sample["text"] for sample in processed_samples]
    
    # Batch tokenization
    logger.info("Tokenizing samples...")
    inputs = tokenizer(
        sample_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=2048  # Adjust based on your model's context size
    )
    
    # Extract activations from model
    logger.info("Extracting activations from model...")
    layer_activations = extract_activations(
        model=model,
        inputs=inputs,
        layers=activation_layers,
        batch_size=batch_size,
        device=device
    )
    
    # Aggregate activations from all layers
    logger.info("Aggregating activations from all layers...")
    aggregated_activations = aggregate_layer_activations(layer_activations)
    
    # Random projection to reduce dimensionality
    logger.info(f"Applying random projection to reduce dimension to {reduced_dim}...")
    projected_activations = random_projection(aggregated_activations, reduced_dim)
    
    # Cluster samples
    logger.info(f"Clustering samples into {num_clusters} clusters...")
    kmeans = cluster_samples(projected_activations, n_clusters=num_clusters, random_state=random_state)
    
    # Select representative samples from each cluster
    logger.info("Selecting representative samples from each cluster...")
    selected_samples = select_representative_samples(processed_samples, projected_activations, kmeans)
    
    logger.info(f"Selected {len(selected_samples)} samples for final calibration dataset")
    
    # Add selection metadata to each sample
    for i, sample in enumerate(selected_samples):
        sample["selection_index"] = i
        sample["selection_method"] = "activation_clustering"
    
    return selected_samples