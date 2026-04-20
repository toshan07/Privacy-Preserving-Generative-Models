# Harpoon

**Harpoon** is a framework for conditional tabular data generation and imputation using diffusion models with manifold-guided updates.  

---

## Repository Structure

The repository is organized to clearly separate training, sampling, and experiments:

- **Training Scripts (`train_*.py`)**  
  Scripts for training models on different baselines and encodings (e.g., Harpoon, TabDDPM, GReaT, RePaint).

- **Sampling Scripts (`sampling_*.py`)**  
  Scripts for generating samples or imputations using trained models. Includes general constraints, OHE/ordinal handling, and manifold-based updates.

- **Experiments (`experiments/`)**  
  Stores experimental setups, batch scripts, and results from different runs.  

- **Datasets (`datasets/`)**  
  Scripts and folders for preparing and managing datasets, including baseline-specific subfolders.

- **Visualization (`visualization/`)**  
  Manifold illustrations.

- **Utility Scripts**  
  - `utils.py` – general-purpose utilities  
  - `dataset.py` – dataset handling and constraints  
  - `diffusion_utils.py` – sampling and model utilities  
  - `generate_mask.py` – missing data masks  
  - `download_and_process.py` – dataset download and preprocessing  

- **LaTeX Generators (`latex_generator_*.py`)**  
  Scripts for generating tables of experiment results.

- **Tubular Neighbourhood Estimation (`tubular_neighbourhood_estimator_*.py`)**  
  Scripts for empirically estimating orthogonality.  



