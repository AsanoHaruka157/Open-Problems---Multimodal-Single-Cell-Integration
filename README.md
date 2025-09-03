# Open-Problems—Multimodal-Single-Cell-Integration

This is my personal solution to the **2022 NeurIPS Kaggle Competition**, and it is still being updated.  
Due to file size limitations, for raw data and the problem statement, please refer to: <https://www.kaggle.com/competitions/open-problems-multimodal>  
For results and processed data, please refer to: <https://drive.google.com/drive/folders/1lZGjnL_C4pkSyAPxxfnYWUPdELiX7_Ay?usp=sharing>

## File Introduction

### `data_processing.ipynb`
This notebook performs the following four steps:
1. Read data from `train_cite_inputs.h5`, `train_cite_targets.h5`, and `test_cite_inputs.h5`.
2. From `train_cite_inputs.h5`, filter out low-variance genes, genes expressed in few cells, and cells that express few genes.
3. Remove cells that express few genes from `train_cite_targets.h5`, and remove genes expressed in few cells from `test_cite_inputs.h5`.
4. Convert the data to a sparse matrix format and store it as `.h5ad` files.

### Other main files
- `metadata.csv`: Metadata for each observation, including cell_id, donor, day, cell type, and other batch-related information; can be used for cross-validation.
- `evaluation_ids.csv`: Used for submitting test-set predictions; based on cell_id and , converts to a globally unique `row_id`.
- `transformer_cite.ipynb`: Builds a Transformer model for the CITE dataset (RNA → Protein) and generates predictions.
- `transformer_multi.ipynb`: Builds a Transformer model for the MULTI dataset (DNA → RNA) and generates predictions.

## Algorithm Overview

### Processing for the CITE dataset
1. Apply z-score normalization to both inputs and targets.
2. Build two groups of Transformer models:  
   - Group 1 uses the mean Pearson correlation across sequences as the loss (higher is better).  
   - Group 2 uses the L2 loss between the predicted and ground-truth matrices.
3. Within each group, train 5 models via cross-validation (5 folds), with splits based on cell type and donor.
4. Ensembling within each group: after training, each of the five models predicts on the test set; the average of their outputs is used as the group prediction.
5. Final ensemble: take a weighted average of the two groups’ predictions (the correlation-based group has a slightly higher weight).

### Processing for the MULTI dataset
Before z-score normalization, apply row-wise normalization separately to the three data groups; the remaining steps are the same as for the CITE pipeline.
