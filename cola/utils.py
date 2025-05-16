# cola/utils.py
"""
Utility functions for the COLA framework.
"""

import os
import logging
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

def setup_logger(name, log_dir, level=logging.INFO):
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name.lower()}.log")
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_json(data, output_path):
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        output_path: Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(input_path):
    """
    Load JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch_process(items, batch_size, process_fn, desc=None):
    """
    Process items in batches with progress bar.
    
    Args:
        items: List of items to process
        batch_size: Batch size
        process_fn: Function to process a batch
        desc: Description for progress bar
        
    Returns:
        List of processed items
    """
    results = []
    
    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    
    return results

class TensorEncoder(json.JSONEncoder):
    """Custom JSON encoder for PyTorch tensors and NumPy arrays."""
    
    def default(self, obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        return super().default(obj)

def evaluate_calibration_samples(
    samples: List[Dict],
    model,
    tokenizer,
    target_capabilities: List[str],
    evaluation_datasets: Dict[str, str],
    batch_size: int = 4,
    device: str = "cuda"
):
    """
    Evaluate selected calibration samples against target capabilities.
    
    Args:
        samples: List of selected calibration samples
        model: The LLM model
        tokenizer: The tokenizer
        target_capabilities: List of capabilities to evaluate
        evaluation_datasets: Dictionary mapping capabilities to evaluation datasets
        batch_size: Batch size for evaluation
        device: Device to run model on
        
    Returns:
        Dictionary with evaluation results
    """
    from datasets import load_dataset
    
    results = {}
    
    for capability in target_capabilities:
        if capability not in evaluation_datasets:
            continue
        
        eval_dataset_name = evaluation_datasets[capability]
        
        try:
            # Load evaluation dataset
            eval_dataset = load_dataset(eval_dataset_name)
            
            # Get first split
            split_name = list(eval_dataset.keys())[0]
            eval_dataset = eval_dataset[split_name]
            
            # Extract text field
            text_field = "text"
            if text_field not in eval_dataset.features:
                for field in eval_dataset.features:
                    if eval_dataset.features[field].dtype == "string":
                        text_field = field
                        break
            
            # Extract sample texts for evaluation
            eval_texts = [item[text_field] for item in eval_dataset[:100]]  # Limit to 100 samples
            
            # Tokenize
            inputs = tokenizer(
                eval_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512  # Shorter for evaluation
            )
            
            # Move to device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Evaluate perplexity
            model.eval()
            perplexities = []
            
            with torch.no_grad():
                for i in range(0, len(eval_texts), batch_size):
                    batch_input_ids = input_ids[i:i+batch_size]
                    batch_attention_mask = attention_mask[i:i+batch_size]
                    
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_input_ids
                    )
                    
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
            
            avg_perplexity = np.mean(perplexities)
            results[capability] = {
                "perplexity": avg_perplexity,
                "dataset": eval_dataset_name
            }
            
        except Exception as e:
            print(f"Error evaluating capability {capability}: {str(e)}")
    
    return results

def generate_sample_statistics(samples: List[Dict]) -> Dict[str, Any]:
    """
    Generate statistics for selected samples.
    
    Args:
        samples: List of selected calibration samples
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_samples": len(samples),
        "datasets": {},
        "avg_length": 0,
        "format_enhanced_count": 0
    }
    
    # Count samples per dataset
    dataset_counts = {}
    for sample in samples:
        dataset_name = sample.get("dataset_name", "unknown")
        if dataset_name not in dataset_counts:
            dataset_counts[dataset_name] = 0
        dataset_counts[dataset_name] += 1
    
    stats["datasets"] = dataset_counts
    
    # Calculate average length
    total_length = 0
    for sample in samples:
        text = sample.get("text", "")
        total_length += len(text.split())
    
    if samples:
        stats["avg_length"] = total_length / len(samples)
    
    # Count format enhanced samples
    for sample in samples:
        if sample.get("format_enhanced", False):
            stats["format_enhanced_count"] += 1
    
    return stats