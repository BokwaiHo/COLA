# cola/dataset_selection.py
"""
Dataset selection module for the COLA framework (Stage 1).
Selects datasets that align with the target deployment domain.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from scipy.special import kl_div
import datasets
from sentence_transformers import SentenceTransformer
from collections import Counter

logger = logging.getLogger("COLA")

def compute_embedding_similarity(dataset1_samples, dataset2_samples, model_name="all-MiniLM-L6-v2"):
    """
    Compute semantic similarity between two datasets using sentence embeddings.
    
    Args:
        dataset1_samples: List of text samples from first dataset
        dataset2_samples: List of text samples from second dataset
        model_name: Sentence transformer model name for computing embeddings
        
    Returns:
        Similarity score between 0 and 1
    """
    # Limit samples to speed up computation
    max_samples = 100
    dataset1_samples = dataset1_samples[:max_samples]
    dataset2_samples = dataset2_samples[:max_samples]
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Compute embeddings
    embeddings1 = model.encode(dataset1_samples, convert_to_tensor=True)
    embeddings2 = model.encode(dataset2_samples, convert_to_tensor=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    
    # Average similarity
    avg_similarity = np.mean(similarity_matrix)
    
    return avg_similarity

def compute_token_distribution_similarity(dataset1_samples, dataset2_samples, tokenizer):
    """
    Compute statistical similarity between token distributions of two datasets.
    
    Args:
        dataset1_samples: List of text samples from first dataset
        dataset2_samples: List of text samples from second dataset
        tokenizer: Tokenizer to use for tokenization
        
    Returns:
        1 - normalized KL divergence (higher means more similar)
    """
    # Tokenize all samples
    tokens1 = [token for sample in dataset1_samples for token in tokenizer.encode(sample)]
    tokens2 = [token for sample in dataset2_samples for token in tokenizer.encode(sample)]
    
    # Count token frequencies
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Get all unique tokens
    all_tokens = set(counter1.keys()) | set(counter2.keys())
    
    # Compute probability distributions
    total1 = sum(counter1.values())
    total2 = sum(counter2.values())
    
    dist1 = np.array([counter1.get(token, 0) / total1 for token in all_tokens])
    dist2 = np.array([counter2.get(token, 0) / total2 for token in all_tokens])
    
    # Smooth distributions to avoid zeros
    epsilon = 1e-10
    dist1 = dist1 + epsilon
    dist2 = dist2 + epsilon
    dist1 = dist1 / np.sum(dist1)
    dist2 = dist2 / np.sum(dist2)
    
    # Compute KL divergence
    kl_12 = entropy(dist1, dist2)
    kl_21 = entropy(dist2, dist1)
    
    # Use symmetric KL
    sym_kl = (kl_12 + kl_21) / 2
    
    # Normalize and convert to similarity (1 - normalized KL)
    max_kl = np.log(len(all_tokens))  # Maximum possible KL
    normalized_kl = sym_kl / max_kl
    
    similarity = 1 - min(normalized_kl, 1.0)  # Ensure it's between 0 and 1
    
    return similarity

def compute_coverage(dataset, capability, reference_datasets, tokenizer, alpha=0.6):
    """
    Compute dataset coverage for a specific capability.
    
    Args:
        dataset: Dataset to evaluate
        capability: Target capability
        reference_datasets: Dictionary mapping capabilities to reference datasets
        tokenizer: Tokenizer for statistical similarity
        alpha: Weight for semantic similarity vs statistical similarity (0-1)
        
    Returns:
        Coverage score for the dataset on the given capability
    """
    if capability not in reference_datasets:
        logger.warning(f"No reference dataset available for capability: {capability}")
        return 0.0
    
    # Get samples from dataset
    dataset_samples = get_dataset_samples(dataset)
    
    # Get samples from reference dataset for this capability
    reference_samples = get_dataset_samples(reference_datasets[capability])
    
    # Compute semantic similarity
    semantic_sim = compute_embedding_similarity(dataset_samples, reference_samples)
    
    # Compute statistical similarity
    statistical_sim = compute_token_distribution_similarity(dataset_samples, reference_samples, tokenizer)
    
    # Combine similarities
    coverage = alpha * semantic_sim + (1 - alpha) * statistical_sim
    
    return coverage

def get_dataset_samples(dataset_name, num_samples=100, seed=42):
    """
    Load samples from a dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        num_samples: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of text samples from the dataset
    """
    try:
        # Load dataset from Hugging Face datasets
        dataset = datasets.load_dataset(dataset_name)
        
        # Get the first split (usually train)
        split_name = list(dataset.keys())[0]
        dataset = dataset[split_name]
        
        # Get text field (assuming it's called "text" - adapt if needed)
        text_field = "text"
        if text_field not in dataset.features:
            # Try to find a text field
            for field in dataset.features:
                if isinstance(dataset.features[field], datasets.features.Value) and \
                   dataset.features[field].dtype == "string":
                    text_field = field
                    break
        
        # Sample from dataset
        np.random.seed(seed)
        if len(dataset) > num_samples:
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            samples = [dataset[i][text_field] for i in indices]
        else:
            samples = [item[text_field] for item in dataset]
        
        return samples
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        return []

def select_datasets(
    available_datasets: List[str],
    target_capabilities: List[str],
    capability_weights: Dict[str, float],
    reference_datasets: Optional[Dict[str, str]] = None,
    tokenizer = None,
    alpha: float = 0.6,
    max_datasets: int = 5
) -> List[Dict]:
    """
    Select optimal datasets for calibration based on capability coverage.
    
    Args:
        available_datasets: List of available dataset names
        target_capabilities: List of capabilities to preserve
        capability_weights: Dictionary mapping capabilities to their importance weights
        reference_datasets: Dictionary mapping capabilities to reference datasets
        tokenizer: Tokenizer for statistical similarity
        alpha: Weight for semantic similarity vs statistical similarity (0-1)
        max_datasets: Maximum number of datasets to select
        
    Returns:
        List of selected datasets with metadata
    """
    # If reference datasets not provided, use defaults
    if reference_datasets is None:
        reference_datasets = {
            "commonsense": "tau/commonsense_qa",
            "math": "allenai/math_qa",
            "code": "lissadesu/code_qa_updated",
            "general": "wikitext",
            "multilingual": "Anthropic/multilingual-evaluations"
        }
    
    # Ensure all target capabilities have reference datasets
    for capability in target_capabilities:
        if capability not in reference_datasets:
            logger.warning(f"No reference dataset for capability: {capability}. Using 'general' reference.")
            reference_datasets[capability] = reference_datasets["general"]
    
    # Compute coverage for each dataset and capability
    dataset_scores = []
    
    for dataset_name in available_datasets:
        # Skip if dataset is a reference dataset to avoid bias
        if dataset_name in reference_datasets.values():
            logger.info(f"Skipping reference dataset: {dataset_name}")
            continue
        
        # Compute weighted coverage across capabilities
        total_score = 0.0
        capability_scores = {}
        
        for capability in target_capabilities:
            score = compute_coverage(
                dataset_name, 
                capability, 
                reference_datasets, 
                tokenizer, 
                alpha
            )
            capability_scores[capability] = score
            total_score += score * capability_weights.get(capability, 1.0)
        
        dataset_scores.append({
            "name": dataset_name,
            "total_score": total_score,
            "capability_scores": capability_scores
        })
    
    # Sort datasets by total score in descending order
    dataset_scores.sort(key=lambda x: x["total_score"], reverse=True)
    
    # Select top datasets up to max_datasets
    selected_datasets = dataset_scores[:max_datasets]
    
    return selected_datasets