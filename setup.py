# setup.py
from setuptools import setup, find_packages

setup(
    name="cola-framework",
    version="0.1.0",
    description="COLA: Curating Optimal LLM compression cAlibration data",
    author="Anonymous",
    author_email="anonymous@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "datasets>=2.4.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)