# MIMIC-CXR Thesis Benchmark

This repository contains the codebase and LaTeX thesis files for evaluating text, image, and multimodal deep learning approaches on the MIMIC-CXR dataset.

## Project Structure
- `notebooks/`: Contains the Jupyter Notebooks for EDA and benchmark training.
  - `Master_MIMIC_CXR_Thesis_Benchmark.ipynb`: The main notebook for training the 3 paradigms (ClinicalBERT, DenseNet121, Multimodal Fusion).
  - `EDA_and_Data_Prep.ipynb`: Notebook for initial data exploration.
- `thesis/`: Contains the LaTeX source code for the thesis.
- `data/test/`: Contains a small dummy dataset for local code testing.

## Execution
The `Master_MIMIC_CXR_Thesis_Benchmark.ipynb` is designed to be executed on Google Colab for access to GPUs and high-speed dataset downloads.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/Master_MIMIC_CXR_Thesis_Benchmark.ipynb)

*Note: The Colab link will work after you push this repository to GitHub.*
