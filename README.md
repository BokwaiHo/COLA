# COLA: Curating Optimal LLM compression cAlibration data

Implementation of the COLA framework proposed in the paper "Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization".

## Overview

COLA is a three-stage framework for curating high-quality calibration data to preserve LLM capabilities during compression:

1. **Dataset Selection (Domain Correspondence)**: Selects datasets that align with the target deployment domain.
2. **Dataset Processing (Compositional Properties)**: Optimizes the compositional properties of selected datasets (sequence length, format, etc.).
3. **Sample Selection (Representativeness and Diversity in Activation Space)**: Selects samples that maximize representativeness and diversity in the model's activation space.

## Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/COLA-7D2C
cd COLA

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cola import COLA

# Load LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

# Initialize COLA framework
cola = COLA(
    model=model,
    tokenizer=tokenizer,
    available_datasets=["wikitext", "c4", "slimpajama-200k"],
    target_capabilities=["commonsense", "math", "code"],
    output_dir="./cola_output"
)

# Run COLA to generate calibration data
calibration_samples = cola.run(
    num_samples=128,          # Number of samples to select
    sequence_length=2048,     # Target sequence length
)
```

### Command-Line Example

You can also use the provided example script:

```bash
python run_cola.py \
    --model_name_or_path meta-llama/Llama-3-8b \
    --output_dir ./cola_output \
    --num_samples 128 \
    --sequence_length 2048 \
    --target_capabilities commonsense math code \
    --datasets wikitext c4 slimpajama-200k \
    --deployment_type general
```

For targeted deployment (focusing on a specific capability):

```bash
python run_cola.py \
    --model_name_or_path meta-llama/Llama-3-8b \
    --output_dir ./cola_output \
    --num_samples 128 \
    --sequence_length 2048 \
    --target_capabilities commonsense math code \
    --datasets wikitext c4 slimpajama-200k \
    --deployment_type targeted \
    --targeted_capability math
```

## Framework Details

### Stage 1: Dataset Selection

This stage focuses on selecting datasets that align with the target deployment domain:

- Analyzes whether the compressed model is intended for general-purpose use or specialized tasks
- Selects a balanced mix of pre-training datasets for general-purpose deployment
- Prioritizes domain-matched datasets for targeted deployment
- Focuses on language alignment, subject coverage, and reasoning difficulty

### Stage 2: Dataset Processing

This stage optimizes the compositional properties of the selected datasets:

- Optimizes sequence length (typically 2048 tokens for most methods)
- Enhances format by converting to Q&A format with explicit reasoning chains
- Filters out low-quality samples that could negatively impact compression

### Stage 3: Sample Selection

This stage selects individual samples to maximize representativeness and diversity in activation space:

- Extracts layer-wise activations from the uncompressed model
- Applies dimensionality reduction using random projection
- Clusters samples in the activation space using k-means
- Selects representative samples from each cluster

## Integration with LLM Compression Methods

COLA is designed to be compatible with various post-training compression methods:

### For Pruning Methods:
- SparseGPT
- Wanda
- LLM-Pruner
- RIA

### For Quantization Methods:
- GPTQ
- AWQ
- SmoothQuant
- FlatQuant

To use COLA with these methods, simply generate the calibration data using COLA, then use it as input for your chosen compression method.

## Example Output

The calibration samples produced by COLA are saved in JSON format:

```json
[
  {
    "text": "Question: How does photosynthesis work?\n\nReasoning:\nPhotosynthesis is the process used by plants, algae and certain bacteria to convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. These are synthesized from carbon dioxide and water.\n\nThe process occurs in multiple steps:\n1. Light energy is absorbed by chlorophyll in the chloroplasts\n2. This energy is used to split water molecules, releasing oxygen\n3. The hydrogen from water and carbon dioxide from the air are used to form glucose\n4. Oxygen is released as a byproduct\n\nThe overall equation is:\n6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\nAnswer: Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen. Chlorophyll captures light energy, which powers chemical reactions that split water and reduce carbon dioxide to create sugar molecules, releasing oxygen as a byproduct.",
    "dataset_name": "c4",
    "capability_scores": {
      "commonsense": 0.85,
      "math": 0.42,
      "code": 0.31
    },
    "format_enhanced": true,
    "selection_index": 14,
    "selection_method": "activation_clustering"
  },
  ...
]
```

## Optimal Calibration Data Characteristics

Based on the paper's findings, optimal calibration data for capability preservation should:

1. **Have representative activation patterns** for the target domain
2. **Maintain diversity in activation space** to cover the model's full capabilities
3. **Include explicit reasoning chains** for preserving reasoning capabilities
4. **Have appropriate sequence length** (typically 2048 tokens for most methods)
5. **Use domain-matched data** for targeted deployment scenarios
6. **Include mixed difficulty levels** for a balance of specialized and general performance


## License

MIT License