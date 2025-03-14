# Multimodal Cardiac Diagnosis

## Datasets

### Data Sources
1. **MRI Dataset**: [Automated Cardiac Diagnosis Challenge (ACDC)](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

2. **ECG Dataset**: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)

### Usage
Please rename the dataset folders into MRI and ECG, and move them into the `data` folder. The file structure should look like this:

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

## Requirements

Please install the dependencies from `requirements.txt` using pip:
```bash
pip install -r requirements.txt
```
