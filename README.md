# MM-IMDb Multimodal Movie Genre Classification

## 1. Project Overview

This project implements a multimodal, multi-label movie genre classification system using the MM-IMDb dataset. The study investigates the use of visual and textual information for predicting movie genres through four classification approaches:

- **Poster-Only:** Uses precomputed VGG visual features extracted from movie posters.
- **Plot-Only:** Uses 768-dimensional textual embeddings generated using DistilBERT.
- **Early Fusion:** Combines visual and textual features before classification.
- **Late Fusion:** Combines prediction probabilities from the independently trained Poster-Only and Plot-Only models.

The objective is to evaluate and compare the performance of unimodal and multimodal approaches across 23 movie genres.

## 2. Dataset

The project uses the MM-IMDb dataset.

**Dataset source:** [MM-IMDb Dataset – Kaggle](https://www.kaggle.com/datasets/johnarevalo/mmimdb)

**Dataset characteristics:**
- Total movie records: 25,959
- Number of genre categories: 23
- Visual representation: 4,096-dimensional precomputed VGG features
- Textual representation: 768-dimensional DistilBERT embeddings
- Target: Multi-label movie genre classification

The dataset is not included in this repository because of its large size. It must be downloaded separately from Kaggle.

## 3. Models Implemented

| Model | Input Representation | Architecture |
|---|---|---|
| Poster-Only | 4,096-dimensional VGG features | 4096 → 512 → 23 |
| Plot-Only | 768-dimensional DistilBERT embeddings | 768 → 256 → 23 |
| Early Fusion | Concatenated 4,864-dimensional features | 4864 → 512 → 23 |
| Late Fusion | Poster and plot prediction probabilities | Weighted probability combination |

## 4. Technologies and Libraries

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- h5py
- Joblib
- Iterative-stratification
- Google Colab

## 5. Repository Contents

```text
mm-imdb-multimodal-genre-classification/
│
├── MM_IMDb_Individual_Implementation.ipynb
│   └── Complete project implementation
│
└── README.md
    └── Project description and reproduction instructions
```

## 6. Requirements and Environment Setup

The project was implemented using Google Colab with Python and PyTorch.

Open the notebook in Google Colab. Install the required packages if they are not already available:

```python
!pip install -q kaggle transformers h5py joblib scikit-learn iterative-stratification
```

A GPU-enabled Colab runtime is recommended for model training and DistilBERT embedding extraction.

## 7. Dataset Setup

1. Create or sign in to a Kaggle account.
2. Download the MM-IMDb dataset using the Kaggle link provided above.
3. Configure Kaggle API authentication in Google Colab.
4. Mount Google Drive when prompted by the notebook.
5. Download and extract the dataset into the project data directory specified in the notebook.

The notebook uses the following Google Drive project structure:

```text
MM_IMDb_Project/
│
├── data/
│   ├── metadata.npy
│   └── multimodal_imdb.hdf5
│
└── results/
```

The dataset and generated model outputs are stored separately from the GitHub repository.

## 8. Execution Instructions

1. Open `MM_IMDb_Individual_Implementation.ipynb` from this repository in Google Colab.
2. Configure the required Kaggle credentials.
3. Execute the notebook cells in their original order.
4. Complete dataset loading, inspection and data quality assessment.
5. Perform iterative multilabel stratification to create training, validation and testing subsets.
6. Prepare visual features and extract textual embeddings using DistilBERT.
7. Train and evaluate the Poster-Only model.
8. Train and evaluate the Plot-Only model.
9. Implement and evaluate the Early Fusion model.
10. Optimise the Late Fusion weight and classification threshold using validation data.
11. Perform final test-set evaluation and compare all four approaches.
12. Generate the performance visualisations and genre-level error analysis.

**Note:** The notebook contains Google Drive mounting and project-path configuration cells. Ensure the dataset and results directories are available at the paths specified in the notebook before executing the relevant sections.

## 9. Evaluation Metrics

The classification approaches are evaluated using:

- Micro F1-score
- Macro F1-score
- Micro Precision
- Micro Recall
- Macro Average Precision
- Micro Average Precision
- Hamming Loss

Threshold optimisation and late-fusion weight selection are performed using validation data, while final model comparisons are reported on the test dataset.

## 10. Experimental Results

The Late Fusion model achieved the following test results:

| Metric | Result |
|---|---:|
| Micro F1-score | 0.5663 |
| Macro F1-score | 0.5506 |
| Micro Precision | 0.5930 |
| Micro Recall | 0.5418 |
| Micro Average Precision | 0.5920 |
| Hamming Loss | 0.0901 |

Late Fusion achieved the highest overall Micro F1-score and Macro F1-score among the four evaluated approaches. Although Late Fusion achieved the best Micro F1-score and Macro F1-score, the Plot-Only model achieved the highest Macro Average Precision (0.6055), demonstrating the effectiveness of textual features in ranking relevant genres.

## 11. Reproducibility Notes

- The MM-IMDb dataset must be downloaded separately from Kaggle.
- Google Colab is used as the execution environment.
- Google Drive is used for storing prepared features, model checkpoints and experimental results.
- Kaggle authentication is required for dataset download through the notebook.
- Execute the notebook cells in their original order and configure the required file paths.
- Results may vary slightly depending on the execution environment and random initialisation.

