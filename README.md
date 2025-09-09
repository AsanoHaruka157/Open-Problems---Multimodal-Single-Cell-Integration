# Open-Problems—Multimodal-Single-Cell-Integration

This is my personal solution to the **2022 NeurIPS Kaggle Competition**, and it is still being updated.  
Due to file size limitations, for raw data and the problem statement, please refer to: <https://www.kaggle.com/competitions/open-problems-multimodal>  
For results and processed data, please kindly run the Jupyter Notebooks.

---

## File Introduction

### Common
- **`data_processing.ipynb`**  
  This notebook performs the following four steps:  
  1) Read data from `train_cite_inputs.h5`, `train_cite_targets.h5`, and `test_cite_inputs.h5`.  
  2) From `train_cite_inputs.h5`, filter out low-variance features, features expressed in few cells, and cells that express few features.  
  3) Remove cells that express few features from `train_cite_targets.h5`, and remove features expressed in few cells from `test_cite_inputs.h5`.  
  4) Convert the data to a sparse matrix format and store it as `.h5ad` files.

- **`metadata.csv`**: Metadata for each observation, including `cell_id`, donor, day, cell type, and other batch-related information; can be used for cross-validation.  
- **`evaluation_ids.csv`**: Used for submitting test-set predictions; based on `cell_id` and `gene_id`, converts to a globally unique `row_id`.

---

### `transformer_cite/multi.ipynb`
- **`transformer_cite.ipynb`**: Builds a Transformer model for the CITE dataset (RNA → Protein) and generates predictions.  
- **`transformer_multi.ipynb`**: Builds a Transformer model for the MULTI dataset (DNA → RNA) and generates predictions.

### `transformer_vae_cite/multi.ipynb`
- **`transformer_vae_cite.ipynb`**: VAE-enhanced Transformer pipeline for CITE (RNA → Protein).  
- **`transformer_vae_multi.ipynb`**: VAE-enhanced Transformer pipeline for MULTI (DNA → RNA).
**The difference between codes for cite and multi is the multi codes loads and processes data with chunks due to the size of dataset.**

### `fnn_svd.ipynb`
- Applied a SVD+FNN pipeline for both CITE and MULTI dataset.

### `fnn_umap_cite/multi.ipynb`
- **`fnn_umap_cite.ipynb`**: Applied SVD-UMAP-FNN pipeline for input X and SVD-FNN for target y.
- **`fnn_umap_multi.ipynb`**: Applied SVD-UMAP-FNN pipeline for both input X and target y.
---

## Algorithm Overview

### Pipeline1: Transformer

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

The final score of this pipeline is 0.541308
---

### Pipeline2: VAE Encoder-Transformer-VAE Decoder

1. **Preprocessing**  
   - For the **input** matrix: divide each column by its **median**.  
   - For the **target** matrix: divide each column by its **median**, then apply **log1p** transform, and finally perform **per-column L2 normalization**.

2. **Merge feature engineering(VAE) with main training(Transformer)**  
   - Replace the former PCA/SVD/variance-threshold dimensionality reduction with a **VAE encoder**.  
   - Use a **Transformer** as the main predictor.  
   - Use the **VAE decoder** to map features back to the target dimensionality.

3. **Loss Design**  
   Total loss = MSE loss - Pearson Correlation Mean + KL Loss

4. **Ensembling**  
   - 5-fold cross-validation (same split strategy as legacy).  
   - The **final prediction** is the **mean** of the 5 models’ test predictions.

The final score of this pipeline is 0.489437
---

### Pipeline 3: SVD+FNN

1.  **Normalization**
    * For **both the input (X) and target (y) matrices**, a Centered Log-Ratio (CLR) transformation is applied to each row vector. For a given row vector **x** with *D* features, the transformation for each component $x_i$ is defined as:
$$
\text{CLR}(x_i) = \log(x_i + p) - \frac{1}{D}\sum_{j=1}^{D} \log(x_j + p)
$$
    where *p* is the pseudocount (typically 1).

2.  **Dimensionality Reduction**
    * **Input (X)**: `TruncatedSVD` with 128 components is fitted on the CLR-transformed training set `X`. The same fitted SVD model is then used to transform the test set `X`.
    * **Target (y)**: `TruncatedSVD` with 128 components is fitted on the CLR-transformed training set `y`.

3.  **Model Architecture**
    * A Feedforward Neural Network (FNN), also known as a Multi-Layer Perceptron (MLP), is used as the predictive model. It takes the 128-dimensional input vector and outputs a 128-dimensional prediction vector.

4.  **Loss Design**
    * The loss is calculated in the 128-dimensional latent space created by SVD.
    * The total loss is a combination of two metrics:
        * **Total Loss = MSE Loss + Negative Pearson Correlation**

    > **Note on Inconsistency**: Your original description included "KL Loss". KL Divergence loss is a component specific to Variational Autoencoders (VAEs) used to regularize the latent space distribution. Since this pipeline uses a standard FNN, which is not a generative model and does not produce a latent distribution (`mu`, `log_var`), the KL Loss is not applicable and has been omitted from this description.

5.  **Ensembling and Prediction**
    * The model is trained using a 5-fold cross-validation strategy.
    * For the final test set prediction:
        1.  Predictions are first made in the 128-dimensional latent space for each of the 5 models.
        2.  These 5 latent-space prediction vectors are **averaged** to create a single ensembled prediction vector.
        3.  Finally, the `inverse_transform` method of the fitted target SVD model (`svd_y`) is applied to this averaged vector to project it back into the original high-dimensional target space.

### Pipeline 4: SVD+UMAP+FNN

1.  **Normalization**
    * For **both the input (X) and target (y) matrices**, a Centered Log-Ratio (CLR) transformation is applied to each row vector. The formula is:
$$
\text{CLR}(x_i) = \log(x_i + p) - \frac{1}{D}\sum_{j=1}^{D} \log(x_j + p)
$$
    where *p* is the pseudocount (typically 1).

2.  **Dimensionality Reduction**
    * **Input (X)**: A two-step process is applied to the CLR-transformed `X`.
        1.  `TruncatedSVD` is used for pre-reduction to 256 components.
        2.  `UMAP` is then applied to the SVD-reduced data to further reduce the dimensionality to 128 non-linear components.
        3.  The same fitted SVD and UMAP mappers are applied to the test set.
    * **Target (y)**: The dimensionality reduction process for the target matrix depends on the dataset:
        * For the **CITE dataset**, a single `TruncatedSVD` is applied to reduce the CLR-transformed `y` to 128 components.
        * For the **Multiome dataset**, a two-step `SVD (256 components) -> UMAP (128 components)` process is applied to the CLR-transformed `y`.

3.  **Model Architecture**
    * A Feedforward Neural Network (FNN/MLP) is used. It maps the 128-dimensional input from the `X` pipeline to a 128-dimensional output, which corresponds to the latent space of the `y` pipeline.

4.  **Loss Design & Validation Metric**
    * **Loss Function**: The loss is calculated in the 128-dimensional latent space.
        * **Total Loss = MSE Loss + Negative Pearson Correlation**
    * **Validation Metric**: The primary validation metric is the **Pearson correlation calculated in the original (CLR-transformed) space**. To compute this metric during validation:
        1.  The model makes a prediction in the 128-dim latent space.
        2.  This prediction is projected back to the high-dimensional CLR space using the appropriate inverse transforms (`.inverse_transform()`).
        3.  The Pearson correlation is then calculated between these reconstructed predictions and the true, high-dimensional CLR-transformed validation targets.

    > **Note on Inconsistency**: As with Pipeline 3, the "KL Loss" mentioned in your original description is not applicable to an FNN model and has been omitted.

5.  **Ensembling and Prediction**
    * The model is trained using a 5-fold cross-validation strategy.
    * For the final test set prediction:
        1.  The 5 models produce predictions in the 128-dimensional latent space.
        2.  These 5 latent-space predictions are **averaged**.
        3.  This averaged vector is projected back to the original high-dimensional space using the inverse transforms of the mappers fitted for the target `y`. The specific steps depend on the dataset:
            * For **CITE**: A single `svd_y.inverse_transform()` is applied.
            * For **Multiome**: A two-step inverse transform is applied: first `umap_y.inverse_transform()`, followed by `svd_y.inverse_transform()`.

### Other Pipelines Tried
We previously tried standalone VAE models, GRU networks, and ResNet, but abandoned them due to poor performance.
