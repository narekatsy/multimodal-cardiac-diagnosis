# Cardiovascular Disease Classification Using Multimodal Deep Learning

## Overview

This project explores the classification of cardiovascular conditions using deep learning models trained on three different modalities: cardiac MRI volumes, ECG signals and patient metadata. The goal is to analyze the feasibility and performance of unimodal (MRI-only and ECG-only) and multimodal (fusion) learning architectures for cardiac disease detection. The work is inspired by real-world challenges in diagnosing heart conditions using heterogeneous sources of medical data.


## EDA Insights

#### ECG Data
- Strong class imbalance across different diagnosis categories
- ECG signals vary in length and frequency, requiring preprocessing (e.g., resampling, normalization)
- Signals contain noise or artifacts, which were filtered during preprocessing

#### MRI Data
- Images have varying exposure and sizes, required resizing and normalization
- Label distribution across the five diagnosis classes is balanced
- Patient metadata shows minimal predictive value for distinguishing between disease classes


## Datasets

This project uses two independent datasets: 
- ECG Data (PTB Diagnostic ECG Database)
- MRI Data (Automated Cardiac Diagnosis Challenge (ACDC))

The datasets are not aligned at the patient level but are used to train modality-specific models for fusion analysis.
These datasets are processed and structured for training as described in the [`DATA.md`](./DATA.md), which includes download instructions and preprocessing details.


## Repository Structure

```bash
multimodal-cardiac-diagnosis
├── checkpoints/                        # Contains saved PyTorch models
├── images/                             # Contains images connected to the project
├── notebooks/
│   ├── eda_ecg.ipynb                   # EDA process of ECG data
│   └── eda_mri.ipynb                   # EDA process of MRI data
│
├── models/
│   ├── older_models/                   # Contains custom encoder and fusion models (not used in training anymore)
│   ├── __init.py__
│   ├── datasets.py                     # Custom dataset loader for PyTorch
│   ├── ecg_encoder.py                  # ResNet-based encoder for ECG signals
│   ├── mri_encoder.py                  # ResNet-based encoder for MRI volumes
│   ├── metadata_encoder.py             # MLP-based encoder for patient metadata  
│   ├── multimodal_fusion_transformer.py    # Transformer-based fusion model for multimodal classification
│   ├── pretrain_ecg.py                 # ECG encoder pre-training
│   ├── pretrain_mri.py                 # MRI encoder pre-training
│   ├── train_fusion.py                 # Multimodal training with MRI, ECG and metadata
│   ├── evaluate_pretrained.py          # Evaluation for ECG or MRI encoder
│   └── evaluate_fusion.py              # Evaluation of multimodal fusion model
│
├── src/
│   ├── __init.py__
│   ├── ecg_splitting.py                # ECG dataset stratified splitting into training/testing
│   ├── ecg_preprocessing.py            # Processing of ECG signals
│   ├── mri_preprocessing.py            # Processing of MRI scans
│   ├── ecg_metadata_preprocessing.py   # Processing of ECG metadata
│   ├── mri_metadata_preprocessing.py   # Processing of MRI metadata
│   └── run_pipeline.bat                # Batch run for ECG, MRI preprocessing tasks  
│
├── data/                           # Raw datasets
│   ├── MRI/
│   └── ECG/
├── split_data/                     # Train and test split datasets
├── processed_data/                 # Preprocessed datasets
│
├── __init__.py                     # Makes the project a Python package
├── .gitignore                      # Specifies files to ignore in git
├── README.md                       # Project documentation
├── DATA.md                         # Data documentation
└── requirements.txt                # Dependencies for the project
```

### Key Components

**Exploratory Data Analysis (EDA)**
- Located in: `notebooks/`
- Notebooks such as eda_ecg.ipynb and eda_mri.ipynb provide insights into the data distributions, class imbalances, and quality of the ECG and MRI datasets. These analyses inform preprocessing and model design choices.

**Data Preprocessing**
- Located in: `src/`
- Scripts like `ecg_preprocessing.py`, `mri_preprocessing.py`, and corresponding metadata preprocessing scripts standardize, clean, and prepare the raw data.
- `ecg_splitting.py` performs a stratified train/test split of ECG data. All processed outputs are saved in the `processed_data/` directory.

**Model Architectures**
- Located in: `models/`
- **Encoders**:
    - `ecg_encoder.py` and `mri_encoder.py`: **ResNet**-based models for extracting deep features from ECG and MRI respectively.
    - `metadata_encoder.py`: A simple MLP to process patient metadata.
- **Fusion Model**:
    - `multimodal_fusion_transformer.py`: Implements a transformer-based architecture for combining features from all modalities.

**Training and Evaluation**
- Located in: `models/`
- `pretrain_ecg.py` and `pretrain_mri.py`: Scripts for unimodal training of ECG and MRI encoders.
- `train_fusion.py`: Trains the multimodal model using MRI, ECG, and metadata.
- `evaluate_pretrained.py` and `evaluate_fusion.py`: Used to evaluate both unimodal and multimodal models on test datasets.

**Model Checkpoints and Outputs**
- Checkpoints of trained models are stored in the `checkpoints/` directory for reuse and evaluation.
- Output metrics and predictions can be logged or saved during evaluation runs.


## Requirements

To run this project smoothly, ensure the following system and environment requirements are met:

- **Disk Space**: At least 8 GB of free disk space to store datasets, preprocessed files, and model checkpoints.

- **Python 3.10+**: Python is required to run the scripts in this project. You can download Python from [here](https://www.python.org/downloads/).

- **Pip Package Manager**: Ensure `pip` is available to install dependencies from `requirements.txt`.

- **CUDA-Enabled GPU**: A system with a CUDA-enabled GPU is **highly recommended** for training the deep learning models efficiently.

- **Datasets**:
    - The MRI and ECG datasets must be downloaded in advance and placed in the `data/` directory.
    - See [`DATA.md`](./DATA.md) for dataset sources, licensing information, and usage instructions.

To install the necessary Python packages, run:


## How to Use

This section provides step-by-step instructions on how others can run your project, from data preparation to model training and evaluation.

**1. Clone the Repository**
Clone this repository to your local machine using:
```bash
git clone https://github.com/narekatsy/multimodal-cardiac-diagnosis.git
cd multimodal-cardiac-diagnosis
```

**2. Install Dependencies**
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

**3. Prepare and Preprocess the Data**
Download the required datasets as mentioned in the [`DATA.md`](./DATA.md) file and follow the instructions.  
Once you have completed the steps, you should see the preprocessed `.npy` files generated in the `processed_data` directory.

**4. Train the Models**
Train the models by running the `models/pretrain_ecg.py` and `models/pretrain_mri.py` files manually. You can adjust the number of epochs or batch size inside the script.  
Next, train the **multimodal fusion model** by running the `models/train_fusion.py` file. The number of epochs and batch size can be adjusted here as well.  
After running the scripts, the trained models will be saved as `.pt` files in the `checkpoints/` folder.

**5. Evaluate the Models**
After training, you can evaluate the models by running either the `models/evaluate_pretrained.py` script for the individual ECG or MRI models, or `models/evaluate_fusion.py` for the multimodal fusion model.
To evaluate the ECG or MRI model specifically, you will need to manually specify which model to load inside the script - this is clearly indicated within the script.


## Outputs and Results

The key outputs of this project include:
- **Model Checkpoints** – Trained models saved in `.pth` format for later evaluation.
- **Performance Metrics** – The evaluation results (accuracy, F1 score, etc.) are printed to the console. It is planned to store them in a `.txt` file in the future.


| Model Type              | Accuracy | Recall   | F1-Score |
|-------------------------|----------|----------|----------|
| Fusion Model ECG input  | 0.3429   | 0.3429   | 0.2987   |
| Fusion Model MRI input  | 0.56     | 0.56     | 0.468    |

| Model Type       | Train Accuracy | Validation Accuracy |
|------------------|----------------|---------------------|
| MRI pre-trained  | 1.0            | 0.8                 |
| ECG pre-trained  | 0.7784         | 0.6364              |
| Fusion Model     | 0.8750         |                     |


## Acknowledgments

I would like to thank [Anna Tshngryan](https://github.com/anna-tshngryan), my project supervisor, for her invaluable guidance throughout this project, suggesting improvements and providing key insights.

Special thanks to [Dr. Habet Madoyan](https://people.aua.am/team_member/habet-madoyan-2/), for providing dedicated computing resources to train the models, greatly accelerating the process.

Finally, I acknowledge the creators of the datasets used in this project for making valuable data available for research purposes.

