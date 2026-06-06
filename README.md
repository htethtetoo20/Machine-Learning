# Seoul Bike Sharing Demand Regression

This repository contains a machine learning workflow for predicting hourly bike rental demand in Seoul using weather and time-based features.

The main work is in a Jupyter notebook that:
- loads and cleans the Seoul bike dataset,
- performs feature engineering,
- trains multiple regression models,
- compares model performance,
- saves trained models for reuse.

## Repository Contents

- `fcc-bikes-regression.ipynb`
	- End-to-end notebook for data prep, training, evaluation, and plots.
- `Data/seoul+bike+sharing+demand/SeoulBikeData.csv`
	- Raw dataset used by the notebook.
- `gbr_bike_model.joblib`
	- Saved Gradient Boosting Regressor model.
- `nn_bike_model.keras`
	- Saved TensorFlow/Keras neural network model.
- `environment.yml`
	- Conda environment definition.
- `requirements.txt`
	- Pip dependency list.

## Problem Statement

Predict `bike_count` (continuous target) from environmental and temporal variables.

In simple terms:
- Input: hour, temperature, humidity, wind speed, rainfall, snowfall, solar radiation, season, holiday/weekend signals, and related features.
- Output: expected number of rented bikes.

## Notebook Workflow

1. Data loading
- Reads `SeoulBikeData.csv`.

2. Feature engineering
- Parses `Date` and extracts:
	- month,
	- day_of_week,
	- is_weekend.
- Encodes categorical features (`Holiday`, `Seasons`).
- Filters non-functioning days.
- Standardizes column naming.

3. Feature selection
- Uses exploratory plots to remove weaker/redundant features.
- Drops:
	- `visibility`,
	- `dew_point_temperature`.

4. Train/validation/test split
- 60% train, 20% validation, 20% test.

5. Model training
- Linear Regression baseline.
- Neural Network (Keras Sequential model with dense layers + early stopping).
- Gradient Boosting Regressor.

6. Evaluation and comparison
- Computes and compares:
	- MSE,
	- MAE,
	- R2.
- Includes residual and prediction visualizations.

7. Model export
- Saves:
	- `nn_bike_model.keras`,
	- `gbr_bike_model.joblib`.

## Models in This Repo

- Linear Regression
	- Fast baseline for a linear relationship check.

- Neural Network
	- Uses learned non-linear feature interactions.
	- Trained with early stopping using validation loss.

- Gradient Boosting Regressor
	- Tree ensemble that incrementally fits residuals.
	- Tracks staged validation MSE to inspect overfitting behavior.

## Environment Setup

You can use either Conda or Pip.

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bike-regression
```

### Option B: Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Project

```bash
jupyter notebook fcc-bikes-regression.ipynb
```

Then run the notebook cells in order.

## Key Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- imbalanced-learn

## Notes

- A fixed random seed is used in the notebook for better reproducibility.
- Saved model files in the repository allow quick reuse without retraining.

## Data Source

As documented in the notebook:
- Seoul open data source (`data.seoul.go.kr`)
- UCI Machine Learning Repository citation metadata.

