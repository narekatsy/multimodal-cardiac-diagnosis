# Dataset Documentation

## Overview

This project uses two independent datasets: PTB Diagnostic ECG Database and ACDC Dataset (MRI) for cardiac diagnosis classification. The datasets are not aligned at the patient level but are used to train modality-specific models for fusion analysis.


## Datasets Used

### ECG Data (PTB Diagnostic ECG Database)

- **Source**: PhysioNet

- **Link**: https://physionet.org/content/ptbdb/1.0.0/

- **Description**: A collection of 549 high-resolution 15-lead ECGs (12 standard leads together with Frank XYZ leads), including clinical summaries for each record. From one to five ECG records are available for each of the 294 subjects, who include healthy subjects as well as patients with a variety of heart diseases.

- **License**: [Open Data Commons Attribution License v1.0](https://physionet.org/content/ptbdb/view-license/1.0.0/)


### MRI Data (Automated Cardiac Diagnosis Challenge (ACDC))

- **Source**: Medical Segmentation Decathlon / MICCAI Challenge

- **Link**: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

- **Description**: Cardiac MRI images from 150 patients grouped into five diagnostic categories. Provided for segmentation and classification tasks.

- **Access**: Freely available for academic research use; requires citation for any use of the dataset.

---

Both datasets are available at the links mentioned above.  
The ECG dataset is 1.66 GB, and the MRI dataset is 2.28 GB in size.


## Usage
The original datasets come with their own directory name and structure, which may not align with this project's organization and model input requirements. For consistency, please rename the parent folders for `MRI` and `ECG` datasets and restructure them in the project directory as follows:

```bash
multimodal-cardiac-diagnosis
│
├── data/
│   ├── MRI/
│   │   ├── training/
│   │   │   ├── patient001
│   │   │   ├── patient002
│   │   │   └── ...
│   │   └── testing/
│   │   
│   └── ECG/
│       ├── patient001
│       ├── patient002
│       ├── patient003
│       └── ...
...
```


## Preprocessing

The preprocessing pipeline ensures the raw datasets are standardized and suitable for model training. It includes steps for both ECG and MRI data.

### 1. Data splitting

The first step is performed using the `src/ecg_splitting.py` script. The script splits the original data into training and testing sets.
Split files are saved under `split_data/ECG/training/` `split_data/ECG/testing/` folders. The MRI dataset should manually be copied or moved from `data` folder into the `split_data` folder.  
In result, the `split_data` folder should have the following structure:

```bash
multimodal-cardiac-diagnosis
│
├── split_data/
│   ├── MRI/
│   │   ├── training/
│   │   │   ├── patient001
│   │   │   ├── patient002
│   │   │   └── ...
│   │   └── testing/
│   │   
│   └── ECG/
│       ├── training/
│       │   ├── patient001
│       │   ├── patient002
│       │   └── ...
│       └── testing/
...
```

### 2. Preprocessing

Preprocessing scripts handle format conversion and data preparation.  
To run the preprocessing pipeline on Windows, navigate to the project root directory and run the following line in the **Command Prompt**.

```bash
src\run_pipeline.bat
```

After this, the ECG signals, MRI volumes and metadata files are saved into the `processed_data` folder.


## Data Limitations

- No paired multimodal data (e.g., no ECG + MRI from the same patient).
- Dataset sizes are relatively small for deep learning models.
- Labels may be imbalanced across classes.


## References

**Citation for PhysioNet Platform**
> Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., & Stanley, H. E. (2000).  
> PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation, 101*(23), e215–e220.  
> 
> https://doi.org/10.1161/01.CIR.101.23.e215

---

**Citation for the use of ACDC dataset**

> Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., et al. (2018).  
> *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?*  
> IEEE Transactions on Medical Imaging, 37(11), 2514–2525.  
>
> https://doi.org/10.1109/TMI.2018.2837502
