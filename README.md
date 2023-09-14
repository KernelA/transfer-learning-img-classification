# [Kaggle transfer learning competition with small training data](https://www.kaggle.com/competitions/platesv2)

## Description


## Requirements

1. Python 3.10 or higher.

## How to run

### Training

Install dependencies:
```
pip install -r. /requirements.txt
```

Setup [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials)
```
dvc repro train
```

Prediction files:
```
dvc repro prediction
```

### EDA

Dependencies from training
```
pip install -r ./requirements.eda.txt
```

Open [eda.ipynb](./eda.ipynb)

### Explain prediction

Dependencies from training
```
pip install -r ./requirements.captum.txt
```

Open [explain_pred.ipynb](./explain_pred.ipynb)

For development:
```
pip install -r ./requirements.dev.txt
```

