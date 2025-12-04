Antibody–Antigen Binding Free Energy Prediction via Structure–Sequence Multimodal Fusion 

This repository contains the implementation of the ABBP-MSSF model, which predicts antibody-antigen binding affinities using multi-scale structural features. The model leverages graph neural networks, sequence embeddings from ESM-2, and structural mutations processed via FoldX.

## Overview
ABBP-MSSF processes protein structures from datasets like AB-Bind, SKEMPI2, and SAbDab. It extracts sequence embeddings, builds graphs, and trains a predictor for binding affinities (ΔG and ΔΔG). The pipeline includes data preparation, feature extraction, model training, and evaluation.

## Dependencies
- Python 3.8+ (tested with Anaconda environment)
- PyTorch
- PyTorch Geometric
- BioPython
- Pandas, NumPy, tqdm
- ESM (for sequence embeddings)
- FoldX (for structural mutations)


### Acquiring External Tools and Models
- **FoldX**: Used for generating mutant structures  
  Obtain an academic or commercial license from the [FoldX Suite homepage](https://foldxsuite.crg.eu). Download the FoldX executable (e.g., `foldx.exe` for Windows) and place it in the `../FOLDX/` directory relative to the project root. Note: FoldX requires a license; academic users can register for free access.

- **ESM-2 Model**: Used for protein sequence embeddings in `pdb_processor.py`.  
  Download the ESM-2 model checkpoint from:  
  [https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)  
  Place the downloaded `.pt` file in the `esm2_t33_650M_UR50D/` directory.  
  (Note: If you intended a different variant like esm2_t30_150M_UR50D, update the link accordingly and adjust paths in `pdb_processor.py`.)

## Directory Structure and File Descriptions
The project is organized as follows. Key files and directories are described below.

- **data_utils.py**: Utility functions for data processing. Includes graph building from PDB files, feature extraction (e.g., node features, edge attributes), dataset loading, and collation functions for PyTorch DataLoader. Handles both wild-type and mutant data.

- **model.py**: Defines the core model architecture (`AffinityPredictor`). Implements graph encoders (using TransformerConv), attention pooling, fusion heads for ΔG and ΔΔG predictions, and ablation flags (e.g., ABLATE_DDG, FIX_GATE_05).

- **train.py**: Training script. Handles K-fold cross-validation, model training, evaluation metrics (RMSE, Pearson correlation), and saving predictions/models. Supports datasets like AB-Bind, SKEMPI2, and SAbDab.

- **datasets/**: Directory for raw and processed datasets.
  - **CSV/**: CSV files containing metadata (e.g., `abbind.csv` for AB-Bind chains and affinities, `skempi2.csv` for SKEMPI2).
  - **FASTA/**: FASTA sequences and processing script.
    - `abbind_sequences.fasta`, `skempi2_sequences.fasta`, etc.: Extracted sequences.
    - `fasta_deal.py`: Script to extract FASTA sequences from PDB files using BioPython.
  - **PDB/**: Processed PDB structures (after mutations using FoldX for mutant datasets).
    - Subdirectories: `abbind/`, `sabdab/`, `skempi2/` – contain PDB files for each dataset.
  - **PDBSOURCE/**: Source (unprocessed) PDB files for mutations.
    - Subdirectories: `abbind/`, `skempi2/` – raw PDBs before FoldX processing.

- **esm2_t33_650M_UR50D/**: Directory for the ESM-2 model checkpoint (place `esm2_t33_650M_UR50D.pt` here).

- **outdata/**: Output directory for processed data and models.
  - **graph/**: Processed graph data in `.pt` format (e.g., `abbind_graphs.pt` – PyTorch Geometric graphs).
  - **outmodel/**: Trained models (e.g., `abbind_bestmodel.pt` – best checkpoint from training).
  - **seq_embeding/**: Sequence embeddings from ESM-2.
    - `abbind_embedings.pt`, `skempi2_embedings.pt`, etc.: Torch tensors of embeddings.
    - `pdb_processor.py`: Script to process FASTA files and generate embeddings using local ESM-2 model.

## Usage
1. **Prepare Data**:
   - For mutant datasets (e.g., AB-Bind, SKEMPI2): Place raw PDB files in `datasets/PDBSOURCE/`. 
     The mutant datasets are generated using FoldX software, with processed structures placed in `datasets/PDB/`. 
     For mutant datasets, during data preprocessing, both mutant PDB files and original PDB files 
     need to be processed simultaneously to generate corresponding files for subsequent testing.
   - For wild-type datasets (e.g., SAbDab): Directly place the original PDB files in `datasets/PDB/`.
   - Run `fasta_deal.py` to extract FASTA sequences from the PDB files in `datasets/PDB/`.
   - Run `pdb_processor.py` to generate sequence embeddings (requires ESM-2).

2. **Build Graphs**:
   - Use `data_utils.py` functions (called internally by `train.py` or manually).

3. **Train the Model**:

4. - Configurable via script variables (e.g., DATASET, EPOCHS, BATCH_SIZE).
- Outputs models to `outdata/outmodel/` and predictions to `outdata/predict/`.

5**Evaluation**:
- Metrics (RMSE, PearsonR) are logged during training.
- Predictions are saved as CSV in `outdata/predict/{DATASET}/`.
