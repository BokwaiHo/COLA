
# cola/__init__.py
"""
COLA (Curating Optimal LLM compression cAlibration data) framework.
Based on the paper: "Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization"

This framework consists of three stages:
1. Dataset Selection (Domain Correspondence)
2. Dataset Processing (Compositional Properties)
3. Sample Selection (Representativeness and Diversity in Activation Space)
"""

from .main import COLA
from .dataset_selection import select_datasets
from .dataset_processing import process_datasets
from .sample_selection import select_samples
from .utils import setup_logger, save_json, load_json, evaluate_calibration_samples

__version__ = "0.1.0"