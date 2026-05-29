# TeraCyte Home Assignment

This repository contains my solution for the TeraCyte home assignment.  
The project includes the analysis notebook, standalone training and evaluation scripts, data split file, requirements file, and a trained model checkpoint.

## Repository structure

```text
.
├── README.md
├── analysis.ipynb
├── train.py
├── evaluate.py
├── requirements.txt
├── split.csv
└── resnet18_full_image_weights.pt
```

## Files

- `analysis.ipynb`  
  Main exploratory notebook used for data analysis, model development, experiments, and result comparison.  
  **Note:** GitHub may fail to render this notebook because of Colab/Jupyter widget metadata. The notebook is valid and can be viewed by downloading it and opening it locally, or by opening it in Google Colab/Jupyter.

- `train.py`  
  Standalone training script. It trains the selected model configuration and saves the model weights.

- `evaluate.py`  
  Standalone evaluation script. It loads a saved checkpoint and prints the test metrics.

- `split.csv`  
  Train/validation/test split used in the experiments.

- [`resnet18_full_image_weights.pt`](./resnet18_full_image_weights.pt)  
  Trained PyTorch model checkpoint used for evaluation.

- `requirements.txt`  
  Python environment exported from Google Colab. Since this file contains the full Colab environment, it is larger than strictly necessary. A smaller local environment with the core packages listed below should also be sufficient.

## Hardware

The main experiments were run on Google Colab using an NVIDIA T4 GPU.

## Setup

Clone the repository:

```bash
git clone https://github.com/GadMil/TeraCyte_HA.git
cd TeraCyte_HA
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Linux/Mac:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file was exported from Colab, so it may include more packages than needed.  
A minimal environment should include:

```text
numpy
pandas
scikit-learn
matplotlib
torch
torchvision
Pillow
opencv-python
tqdm
```

## Data and path setup

Before running the scripts, make sure the file paths are set correctly.

The scripts expect:

1. A data directory containing the image folder and the metadata file.
2. The `split.csv` file to define the train/validation/test split.
3. A checkpoint path for saving/loading the model weights.

Please open `train.py` and `evaluate.py` and update the path variables at the top of the files according to your local folder structure.

For example:

```python
DATA_DIR = "/path/to/data"
SPLIT_CSV = "split.csv"
CHECKPOINT_PATH = "resnet18_full_image_weights.pt"
```

The most important requirement for reproducing the results is that `DATA_DIR` correctly points to the folder containing the images referenced by `split.csv`.

If the image paths inside `split.csv` are relative paths, then `DATA_DIR` should point to their root folder.  
If the image paths are absolute paths, update them or adjust the loading code accordingly.

## Reproducing the results

### Option 1: Run the notebook

Open and run:

```text
analysis.ipynb
```

This notebook contains the full experimental workflow, including data loading, preprocessing, model experiments, and result comparison.

### Option 2: Train from script

After setting the correct paths in `train.py`, run:

```bash
python train.py
```

This will train the model and save the checkpoint to the configured checkpoint path.

### Option 3: Evaluate from checkpoint

After setting the correct paths in `evaluate.py`, run:

```bash
python evaluate.py
```

This script loads the trained checkpoint:

```text
resnet18_full_image_weights.pt
```

and prints the evaluation metrics for the experiment.

## Model checkpoint

The trained model checkpoint is included in the repository:

```text
resnet18_full_image_weights.pt
```

It can also be accessed directly here:

[resnet18_full_image_weights.pt](./resnet18_full_image_weights.pt)

If downloading separately, place the checkpoint in the repository root directory or update `CHECKPOINT_PATH` in `evaluate.py` to point to its location.

## Notes

- The main model used for the final evaluation is a ResNet18-based image model.
- The assignment was developed in Google Colab, so some paths may need to be changed when running locally.
- The scripts are intended to be simple and reproducible rather than heavily engineered.
- The notebook contains additional analysis and intermediate experiments, while `train.py` and `evaluate.py` provide the runnable training and evaluation flow.
