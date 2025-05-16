# run_cola.py
"""
Example script to use COLA framework for curating calibration data.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cola import COLA

def main():
    parser = argparse.ArgumentParser(description="Run COLA framework for calibration data curation")
    
    # Model and data arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="./cola_output",
                        help="Directory to save calibration data")
    
    # COLA framework arguments
    parser.add_argument("--num_samples", type=int, default=128,
                        help="Number of samples to select for the final calibration dataset")
    parser.add_argument("--sequence_length", type=int, default=2048,
                        help="Target sequence length for processed samples")
    parser.add_argument("--target_capabilities", type=str, nargs="+", 
                        default=["commonsense", "math", "code"],
                        help="List of capabilities to preserve")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["wikitext", "c4", "allenai/c4", "Salesforce/wikitext", "DKYoon/slimpajama-200k"],
                        help="List of available datasets for selection")
    
    # Deployment scenario
    parser.add_argument("--deployment_type", type=str, choices=["general", "targeted"], default="general",
                        help="Deployment scenario: general or targeted")
    parser.add_argument("--targeted_capability", type=str, default=None,
                        help="Target capability for targeted deployment")
    
    # Hardware settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print(f"Loading model {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Configure COLA framework
    if args.deployment_type == "targeted" and args.targeted_capability is not None:
        # Targeted deployment: emphasize specific capability
        target_capabilities = [args.targeted_capability]
        capability_weights = {args.targeted_capability: 1.0}
        
        # Add other capabilities with lower weights
        for cap in args.target_capabilities:
            if cap != args.targeted_capability:
                target_capabilities.append(cap)
                capability_weights[cap] = 0.2  # Lower weight for non-targeted capabilities
    else:
        # General deployment: balanced weights
        target_capabilities = args.target_capabilities
        capability_weights = {cap: 1.0 / len(target_capabilities) for cap in target_capabilities}
    
    # Initialize COLA framework
    cola = COLA(
        model=model,
        tokenizer=tokenizer,
        available_datasets=args.datasets,
        target_capabilities=target_capabilities,
        capability_weights=capability_weights,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Additional parameters for each stage
    stage1_params = {
        "alpha": 0.6,  # Weight for semantic similarity vs statistical similarity
    }
    
    stage2_params = {
        "add_reasoning_chains": True,
        "min_length": 256,
        "filter_low_quality": True
    }
    
    stage3_params = {
        "batch_size": args.batch_size,
    }
    
    # Run COLA framework
    calibration_samples = cola.run(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        stage1_params=stage1_params,
        stage2_params=stage2_params,
        stage3_params=stage3_params
    )
    
    print(f"COLA framework completed. Selected {len(calibration_samples)} samples for calibration.")
    print(f"Calibration data saved to {args.output_dir}")

if __name__ == "__main__":
    main()