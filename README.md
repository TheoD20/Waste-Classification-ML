# Waste Classification ML

This project investigates image-based household waste classification using the Kaggle Garbage Classification dataset with 12 waste categories.

The goal is to compare multiple machine learning approaches on the same dataset, including traditional machine learning and deep learning methods. The project is developed in Python using Jupyter notebooks.

## Project Objective

The objective is to classify waste images into their correct material/category classes and compare how different model families perform under a consistent evaluation protocol.

Model approaches include:

- HOG + Random Forest
- LDA/PCA + SVM
- From Scratch CNN (VGG style)
- MobileNet Transfer Learning

## Group Members

This project is developed as a group, formed by:
- Theo Negrao
- Yuta Matsuzaki
- Marimbay Kadirov
- Gyeonghwan Noh

## Dataset

- Dataset: Kaggle Garbage Classification  
- Classes: 12 garbage/waste categories
- Link: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

The dataset is not stored in this GitHub repository. It can be downloaded locally using the Kaggle API.

The shared train/validation/test split files will be stored at:
```text
data/splits/
```

## Report
The final project report will be added once the analysis and write-up are complete. The report will summarise the dataset, preprocessing, model approaches, evaluation protocol, results, computational cost, and key findings from the comparison between traditional machine learning and deep learning methods.

## Expected local structure:

```text
WasteClassification/
├── random_forest/
│   ├── figures/
│   ├── random_forest_model.ipynb        (main rf model training)
|   ├── random_forest_metrics.json         - results
│   └── random_forest_prediction.csv
├── pca_lda/
│   ├── figures/
│   ├── pca_lda_model.ipynb              (main svg model training)
|   ├── pca_lda_metrics.json               - results
│   └── pca_lda_prediction.csv
├── scratch_cnn/
│   ├── figures/
│   ├── scratch_cnn_model.ipynb          (main scratch cnn model training)
|   ├── scratch_cnn_metrics.json           - results
│   └── scratch_cnn_prediction.csv
├── mobilenet_v3/
│   ├── figures/
│   ├── mobilenet_v3_model.ipynb         (main tranfer training model training)
|   ├── mobilenet_v3_metrics.json          - results
│   └── mobilenet_v3_prediction.csvs
|
├── data_audit_and_split.ipynb           (inspect data and split - train/test/val)
├── final_comparison.ipynb               (compare and analyse final models)
├── final_evaliuation/                   (folder with final evaluation results and figures)
|
├── data/
│   ├── garbage_classification/
│   └── splits/
|
├── shared_utils.py                      (shared functions: load_data(), eval_model(), etc)
├── LICENSE
├── README.md
└── .gitignore
```

## Setup

To download the full raw dataset:

1. Install the required Kaggle API package:
```
python -m pip install --upgrade kaggle
```

2. Download the dataset:

```
mkdir -p data

kaggle datasets download \
  -d mostafaabla/garbage-classification \
  -p data

unzip -q data/garbage-classification.zip -d data

rm data/garbage-classification.zip
```

Another option is to manually download the dataset as a zip file from Kaggle:
https://www.kaggle.com/datasets/mostafaabla/garbage-classification

If downloading manually, extract the dataset so that the final folder path is:
```text
data/garbage_classification/
```