# (BERT) Fine-tuning for Fake News Classification
A side project to practice fine-tuning BERT for sequence classification.

## Data source
The data is downloaded from kaggle: [Click me](https://www.kaggle.com/c/fake-news-pair-classification-challenge)
- Download the data and put the csv files in ./data folder.

## Important Dependencies:
- transformers
- pyyaml
- torch
- numpy
- pandas

## Files:
- `main.py` : Main pipeline of training.
- `train.py`: Prediction function and train_for_one_epoch function.
- `dataset.py`: The dataset class.
- `preprocess.py`: Clear the data.
