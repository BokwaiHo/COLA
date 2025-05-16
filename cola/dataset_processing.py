# cola/dataset_processing.py
"""
Dataset processing module for the COLA framework (Stage 2).
Optimizes the compositional properties of the selected datasets.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any
from transformers import PreTrainedTokenizer

logger = logging.getLogger("COLA")

def tokenize_text(text: str, tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> Dict[str, Any]:
    """
    Tokenize text and prepare for model input.
    
    Args:
        text: Input text to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs
    """
    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    return {
        "text": text,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

def optimize_sequence_length(
    samples: List[Dict],
    tokenizer: PreTrainedTokenizer,
    target_length: int = 2048
) -> List[Dict]:
    """
    Optimize sequence length of samples.
    
    Args:
        samples: List of text samples
        tokenizer: Tokenizer to use
        target_length: Target sequence length
        
    Returns:
        List of processed samples with optimized sequence length
    """
    processed_samples = []
    
    for sample in samples:
        text = sample["text"]
        
        # Check current length
        tokens = tokenizer.encode(text)
        current_length = len(tokens)
        
        if current_length <= target_length:
            # If shorter than target, keep as is
            processed_sample = tokenize_text(text, tokenizer, target_length)
            processed_samples.append(processed_sample)
        else:
            # If longer than target, truncate with some context awareness
            # Try to break at paragraph or sentence boundaries
            paragraphs = text.split("\n\n")
            
            current_text = ""
            current_tokens = 0
            
            for paragraph in paragraphs:
                paragraph_tokens = tokenizer.encode(paragraph)
                paragraph_length = len(paragraph_tokens)
                
                # If adding this paragraph would exceed target length, break
                if current_tokens + paragraph_length > target_length - 2:  # -2 for special tokens
                    break
                
                current_text += paragraph + "\n\n"
                current_tokens += paragraph_length + 1  # +1 for newline
            
            # If we didn't get enough text, take the first target_length tokens
            if current_tokens < target_length // 2:
                current_text = tokenizer.decode(tokens[:target_length - 2])
            
            processed_sample = tokenize_text(current_text, tokenizer, target_length)
            processed_samples.append(processed_sample)
    
    return processed_samples

def enhance_format_with_reasoning(sample: Dict) -> Dict:
    """
    Enhance sample format with explicit reasoning chains if possible.
    
    Args:
        sample: Input sample dictionary
        
    Returns:
        Sample with enhanced format if possible
    """
    text = sample["text"]
    
    # Check if it already has explicit reasoning format
    if "Reasoning:" in text or "Step 1:" in text or "Chain of Thought:" in text:
        return sample
    
    # Patterns to detect implicit reasoning
    qa_pattern = re.search(r"(Question|Q):?([^\n]+).*?(Answer|A):?([^\n]+)", text, re.DOTALL)
    math_pattern = re.search(r"(Problem|Calculate|Solve):?([^\n]+).*?(Solution|Result|Answer):?([^\n]+)", text, re.DOTALL)
    
    if qa_pattern:
        # This looks like a QA pair, try to insert reasoning
        question = qa_pattern.group(2).strip()
        answer = qa_pattern.group(4).strip()
        
        # Look for explanatory text between question and answer
        explanation_match = re.search(f"{re.escape(question)}(.*?){re.escape(answer)}", text, re.DOTALL)
        
        if explanation_match and len(explanation_match.group(1).strip()) > 20:
            # There seems to be an explanation, format it as reasoning
            explanation = explanation_match.group(1).strip()
            
            # Format with explicit reasoning chain
            formatted_text = f"Question: {question}\n\nReasoning:\n{explanation}\n\nAnswer: {answer}"
            
            # Update the sample
            updated_sample = sample.copy()
            updated_sample["text"] = formatted_text
            updated_sample["format_enhanced"] = True
            return updated_sample
    
    elif math_pattern:
        # This looks like a math problem, try to insert reasoning
        problem = math_pattern.group(2).strip()
        solution = math_pattern.group(4).strip()
        
        # Look for steps between problem and solution
        steps_match = re.search(f"{re.escape(problem)}(.*?){re.escape(solution)}", text, re.DOTALL)
        
        if steps_match and len(steps_match.group(1).strip()) > 20:
            # There seems to be solution steps, format it as reasoning
            steps = steps_match.group(1).strip()
            
            # Format with explicit reasoning steps
            formatted_text = f"Problem: {problem}\n\nSolution Steps:\n{steps}\n\nAnswer: {solution}"
            
            # Update the sample
            updated_sample = sample.copy()
            updated_sample["text"] = formatted_text
            updated_sample["format_enhanced"] = True
            return updated_sample
    
    # If we couldn't enhance the format, return the original sample
    return sample

def process_datasets(
    selected_datasets: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    add_reasoning_chains: bool = True,
    min_length: int = 256,
    filter_low_quality: bool = True
) -> List[Dict]:
    """
    Process the selected datasets by optimizing their compositional properties.
    
    Args:
        selected_datasets: List of datasets selected in Stage 1
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        add_reasoning_chains: Whether to add explicit reasoning chains
        min_length: Minimum sequence length to keep
        filter_low_quality: Whether to filter low-quality samples
        
    Returns:
        List of processed samples ready for Stage 3
    """
    all_processed_samples = []
    
    for dataset_info in selected_datasets:
        dataset_name = dataset_info["name"]
        logger.info(f"Processing dataset: {dataset_name}")
        
        try:
            # Get raw samples from the dataset
            from datasets import load_dataset
            
            # Load dataset (assuming it's available in Hugging Face datasets)
            dataset = load_dataset(dataset_name)
            
            # Get the first split (usually train)
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            
            # Convert to samples with text field
            text_field = "text"
            if text_field not in dataset.features:
                # Try to find a text field
                for field in dataset.features:
                    if dataset.features[field].dtype == "string":
                        text_field = field
                        break
            
            raw_samples = [{"text": item[text_field], "source": dataset_name} for item in dataset]
            
            # Filter samples based on length
            if filter_low_quality:
                filtered_samples = []
                for sample in raw_samples:
                    # Check length
                    tokens = tokenizer.encode(sample["text"])
                    if len(tokens) >= min_length:
                        filtered_samples.append(sample)
                
                logger.info(f"Filtered {len(raw_samples) - len(filtered_samples)} samples below min_length")
                raw_samples = filtered_samples
            
            # Optimize sequence length
            length_optimized_samples = optimize_sequence_length(
                raw_samples, tokenizer, max_length
            )
            
            # Enhance format with reasoning chains if requested
            if add_reasoning_chains:
                enhanced_samples = []
                for sample in length_optimized_samples:
                    enhanced_sample = enhance_format_with_reasoning(sample)
                    enhanced_samples.append(enhanced_sample)
                
                # Count enhanced samples
                enhanced_count = sum(1 for s in enhanced_samples if s.get("format_enhanced", False))
                logger.info(f"Enhanced format for {enhanced_count} samples with explicit reasoning chains")
                
                final_samples = enhanced_samples
            else:
                final_samples = length_optimized_samples
            
            # Add dataset metadata to each sample
            for sample in final_samples:
                sample["dataset_name"] = dataset_name
                sample["dataset_score"] = dataset_info["total_score"]
                sample["capability_scores"] = dataset_info["capability_scores"]
            
            all_processed_samples.extend(final_samples)
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
    
    logger.info(f"Total processed samples: {len(all_processed_samples)}")
    return all_processed_samples