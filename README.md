# Open-Problems—Multimodal-Single-Cell-Integration

This is my personal solution to the **2022 NeurIPS Kaggle Competition**, and it is still being updated.  
Due to file size limitations, for raw data and the problem statement, please refer to: <https://www.kaggle.com/competitions/open-problems-multimodal>  
For results and processed data, please refer to: <https://drive.google.com/drive/folders/1lZGjnL_C4pkSyAPxxfnYWUPdELiX7_Ay?usp=sharing>

---

## File Introduction

### Common
- **`data_processing.ipynb`**  
  This notebook performs the following four steps:  
  1) Read data from `train_cite_inputs.h5`, `train_cite_targets.h5`, and `test_cite_inputs.h5`.  
  2) From `train_cite_inputs.h5`, filter out low-variance genes, genes expressed in few cells, and cells that express few genes.  
  3) Remove cells that express few genes from `train_cite_targets.h5`, and remove genes expressed in few cells from `test_cite_inputs.h5`.  
  4) Convert the data to a sparse matrix format and store it as `.h5ad` files.

- **`metadata.csv`**: Metadata for each observation, including `cell_id`, donor, day, cell type, and other batch-related information; can be used for cross-validation.  
- **`evaluation_ids.csv`**: Used for submitting test-set predictions; based on `cell_id` and `gene_id`, converts to a globally unique `row_id`.

---

### Legacy (旧版): `transformer_{name}`
- **`transformer_cite.ipynb`**: Builds a Transformer model for the CITE dataset (RNA → Protein) and generates predictions.  
- **`transformer_multi.ipynb`**: Builds a Transformer model for the MULTI dataset (DNA → RNA) and generates predictions.

### New (新版): `transformer_vae_{name}`
- **`transformer_vae_cite.ipynb`**: VAE-enhanced Transformer pipeline for CITE (RNA → Protein).  
- **`transformer_vae_multi.ipynb`**: VAE-enhanced Transformer pipeline for MULTI (DNA → RNA).

---

## Algorithm Overview

### Legacy pipeline: `transformer_{name}`

**Processing for the CITE dataset**
1. Apply z-score normalization to both inputs and targets.  
2. Build two groups of Transformer models:  
   - *Group 1*: loss = mean Pearson correlation across sequences (higher is better).  
   - *Group 2*: loss = L2 (MSE) between predictions and ground truth.  
3. For each group, train 5 models via cross-validation (5 folds), with splits based on cell type and donor.  
4. **Intra-group ensembling**: each of the five models predicts on the test set; average their outputs as the group prediction.  
5. **Final ensemble**: take a weighted average of the two groups’ predictions (the correlation-based group has slightly higher weight).

**Processing for the MULTI dataset**
- Before z-score normalization, apply row-wise normalization separately to the three data groups; the remaining steps are the same as for the CITE pipeline.

---

### New pipeline: `transformer_vae_{name}`

1. **Preprocessing**  
   - For the **input** matrix: divide each column by its **median**.  
   - For the **target** matrix: divide each column by its **median**, then apply **log1p** transform, and finally perform **per-column L2 normalization**.

2. **Merge feature engineering(VAE) with main training(Transformer)**  
   - Replace the former PCA/SVD/variance-threshold dimensionality reduction with a **VAE encoder**.  
   - Use a **Transformer** as the main predictor.  
   - Use the **VAE decoder** to map features back to the target dimensionality.

3. **Loss Design**  
   \[
   \text{Total Loss} \;=\; \underbrace{\text{MSE}}_{\text{reconstruction/prediction}} \;-\; \underbrace{\text{Row-wise mean Pearson}}_{\text{maximize correlation}} \;+\; \underbrace{\text{KL divergence}}_{\text{VAE regularization}}
   \]

4. **Ensembling**  
   - 5-fold cross-validation (same split strategy as legacy).  
   - The **final prediction** is the **mean** of the 5 models’ test predictions.

---
