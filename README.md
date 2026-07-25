# Predictive Clusterwise Linear Regression via Variable Neighborhood Descent

A Python implementation of a Variable Neighborhood Descent (VND) framework for Clusterwise Linear Regression (CLR). This approach addresses a common limitation in standard CLR: the decoupling of clustering and cluster-prediction. By jointly optimizing cluster assignments for both regression accuracy and cluster-label predictability, the model learns clusters that are not only fit for local linear models but are also easy to recover for unseen data points.

## Methodology

Standard CLR methods optimize clusters solely for regression error, which can result in fragmented clusters that are difficult for a classifier to predict. This framework jointly optimizes a weighted objective function:

`Error = α * Regression_Error + (1 - α) * Classification_Error`

Where:
*   **Regression Error**: Clusterwise Mean Squared Error.
*   **Classification Error**: Cross-entropy loss of the cluster-assignment classifier.

The VND algorithm explores $p$-swap neighborhoods ($N_p(a)$) of the cluster assignment vector. For every candidate assignment, the local regression models and the cluster classifier are retrained, and the new solution is evaluated using the joint objective function. 

## Repository Structure

```text
.
├── clr/                        # Core algorithm implementations
│   ├── models/                 # Model definitions
│   │   ├── base.py             # BaseCLR class with shared math and VND logic
│   │   ├── vnd_recursive.py    # VND using recursive neighborhood search (Logistic & Tree)
│   │   ├── vnd_iterative.py    # VND using iterative flat-loop search with a time deadline
│   │   ├── clr_kipok.py        # Decoupled CLR baseline (Gitman et al.) with inference strategies
│   │   └── prediction.py       # Standalone inference strategies (KNN, centroids, etc.)
│   ├── core/                   # Core alternating minimization logic for baseline CLR
│   ├── utils/                  # Checkpoint manager for long experiments
│   └── metrics.py              # Evaluation metrics (RMSE, MAE, MASE, CE)
├── experiments/                # Scripts for data generation, CV evaluation, and plotting
│   ├── experiment_utils.py     # Shared experiment utilities (datasets, plots, summaries)
│   ├── evaluate_synthetic_cv.py# Runs 5-seed CV on synthetic data and Auto-MPG
│   └── eval_tree_cv_tree_iter_2.py # Checkpointed evaluation script
└── tests/                      # Smoke tests for model verification
```

## Implemented Models

1.  **`CLR_VND_Logistic`**: Joint VND optimization using Logistic Regression for cluster prediction.
2.  **`CLR_VND_Tree`**: Joint VND optimization using a Random Forest classifier for cluster prediction.
3.  **`CLR_VND_TreeIterative`**: A flat-loop, iterative variant of the VND search with a hard time deadline (prevents stack overflow on large datasets).
4.  **`CLR_Kipok`**: A standard decoupled CLR baseline (Gitman et al., 2018) with multiple inference strategies for unseen points (`clrp`, `knn`, `knn_weighted`, `centroid`, `distance`, `rmse_local`, etc.).

## Experiments

The experiments evaluate the methods on five synthetic datasets (DS1-DS5) with varying covariance structures and cluster overlaps, as well as the real-world **Auto-MPG** dataset. 

*   **Protocol**: 80/20 train-test split. 5-fold cross-validation on the training set, repeated over 5 random seeds.
*   **Metrics**: Root Mean Squared Error (RMSE) and Coefficient of Determination ($R^2$).
*   **Parameters**: $K=3$ clusters, neighborhood size $p_{max}=1$, and trade-off parameter $\alpha \in \{1.0, 0.9, 0.75, 0.5, 0.25, 0.1\}$.

## Requirements

*   Python 3.8+
*   `numpy`
*   `scipy`
*   `scikit-learn`
*   `matplotlib`
*   `pandas`

## Usage

To run the full experimental evaluation (matching the methodology of the paper) and generate the grand summary table:

```bash
python experiments/evaluate_synthetic_cv.py
```

To run the checkpointed evaluation script (saves progress incrementally to avoid losing results on long runs):

```bash
python experiments/eval_tree_cv_tree_iter_2.py
```

To verify that all models are wired up correctly:

```bash
python tests/smoke_test.py
```