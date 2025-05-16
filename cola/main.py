# cola/main.py
"""
Main module for COLA (Curating Optimal LLM compression cAlibration data) framework.
Based on the paper: "Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization"
"""

import os
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Callable
from .dataset_selection import select_datasets
from .dataset_processing import process_datasets
from .sample_selection import select_samples
from .utils import setup_logger

class COLA:
    """
    COLA (Curating Optimal LLM compression cAlibration data) framework implementation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        available_datasets: List[str],
        target_capabilities: List[str],
        capability_weights: Optional[Dict[str, float]] = None,
        output_dir: str = "cola_output",
        seed: int = 42,
        device: str = None,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize COLA framework.
        
        Args:
            model: The original uncompressed LLM model
            tokenizer: The tokenizer for the model
            available_datasets: List of available dataset names for selection
            target_capabilities: List of capabilities to preserve (e.g., ["commonsense", "math", "code"])
            capability_weights: Dictionary mapping capabilities to their importance weights
            output_dir: Directory to save outputs
            seed: Random seed for reproducibility
            device: Device to run the model on (e.g., "cuda:0", "cpu")
            logging_level: Logging level
        """
        self.model = model
        self.tokenizer = tokenizer
        self.available_datasets = available_datasets
        self.target_capabilities = target_capabilities
        
        # Set default equal weights if not provided
        if capability_weights is None:
            self.capability_weights = {cap: 1.0 / len(target_capabilities) for cap in target_capabilities}
        else:
            self.capability_weights = capability_weights
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set random seed
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Setup logger
        self.logger = setup_logger("COLA", output_dir, logging_level)
        self.logger.info(f"Initializing COLA framework with capabilities: {target_capabilities}")
        self.logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def run(
        self,
        num_samples: int = 128,
        sequence_length: int = 2048,
        stage1_params: Optional[Dict] = None,
        stage2_params: Optional[Dict] = None,
        stage3_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Run the full COLA framework pipeline.
        
        Args:
            num_samples: Number of samples to select for the final calibration dataset
            sequence_length: Target sequence length for processed samples
            stage1_params: Additional parameters for dataset selection stage
            stage2_params: Additional parameters for dataset processing stage
            stage3_params: Additional parameters for sample selection stage
            
        Returns:
            List of selected calibration samples
        """
        self.logger.info("Starting COLA framework pipeline")
        
        # Default parameters for each stage
        default_stage1_params = {
            "alpha": 0.6,  # Weight for semantic similarity vs statistical similarity
        }
        
        default_stage2_params = {
            "add_reasoning_chains": True,
            "max_length": sequence_length,
        }
        
        default_stage3_params = {
            "num_clusters": num_samples,
            "reduced_dim": 64,
            "activation_layers": None,  # Use all layers by default
        }
        
        # Update with user-provided parameters
        if stage1_params:
            default_stage1_params.update(stage1_params)
        
        if stage2_params:
            default_stage2_params.update(stage2_params)
        
        if stage3_params:
            default_stage3_params.update(stage3_params)
        
        # Stage 1: Dataset Selection
        self.logger.info("Stage 1: Dataset Selection")
        selected_datasets = select_datasets(
            available_datasets=self.available_datasets,
            target_capabilities=self.target_capabilities,
            capability_weights=self.capability_weights,
            **default_stage1_params
        )
        self.logger.info(f"Selected datasets: {selected_datasets}")
        
        # Stage 2: Dataset Processing
        self.logger.info("Stage 2: Dataset Processing")
        processed_samples = process_datasets(
            selected_datasets=selected_datasets,
            tokenizer=self.tokenizer,
            **default_stage2_params
        )
        self.logger.info(f"Processed {len(processed_samples)} samples")
        
        # Stage 3: Sample Selection
        self.logger.info("Stage 3: Sample Selection")
        final_samples = select_samples(
            processed_samples=processed_samples,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            **default_stage3_params
        )
        self.logger.info(f"Selected {len(final_samples)} final samples")
        
        # Save the final samples
        self._save_samples(final_samples)
        
        return final_samples
    
    def _save_samples(self, samples: List[Dict]) -> None:
        """
        Save the selected calibration samples to disk.
        
        Args:
            samples: List of selected calibration samples
        """
        import json
        
        samples_path = os.path.join(self.output_dir, "calibration_samples.json")
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(samples)} calibration samples to {samples_path}")