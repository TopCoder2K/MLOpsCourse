# MLOpsCourse

This repository is dedicated to the easy-to-understand practice of the standard MLOps
techniques.

## Task Description

The "Bike Rentals" dataset is used for scripts in this repository. This dataset contains
daily counts of rented bicycles from the bicycle rental company
[Capital-Bikeshare](https://capitalbikeshare.com/) in Washington D.C., along with weather
and seasonal information. The goal is to predict how many bikes will be rented depending
on the weather and the day. The train split info is

```
Index: 8645 entries, 0 to 8644
Data columns (total 11 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   season      8645 non-null   category
 1   month       8645 non-null   int64
 2   hour        8645 non-null   int64
 3   holiday     8645 non-null   category
 4   weekday     8645 non-null   int64
 5   workingday  8645 non-null   category
 6   weather     8645 non-null   category
 7   temp        8645 non-null   float64
 8   feel_temp   8645 non-null   float64
 9   humidity    8645 non-null   float64
 10  windspeed   8645 non-null   float64
```

## Setup

First of all, clone the repository

```
git clone https://github.com/TopCoder2K/mlops-course.git
```

To setup only the necessary dependencies, run the following:

```
poetry install --without dev
```

If you want to use `pre-commit` and `dvc`, install all the dependencies:

```
poetry install
```

## Fetching the data

To fetch the preprocessed train and test splits of the dataset, run:

```
poetry run dvc pull
```

The command should download two .csv files from my GDrive and place them inside the
`mlopscourse/data/` directory.

## Running experiments

### Training

If you want to train the chosen model and save it afterwards, place its configuration file
in the `mlopscourse/configs` directory and run:

```
poetry run python3 commands.py train --config_name [config_name_without_extension]
```

The available models are `rf` (Random Forest from the `scikit-learn` library) and `cb`
(Yandex's CatBoost), so an example with the CatBoost would be the following:

```
poetry run python3 commands.py train --config_name cb_config
```

### Evaluation

If you want to infer a previously trained model, make sure you've placed the checkpoint in
`checkpoints/` and the configuration file in `mlopscourse/configs` then run

```
poetry run python3 commands.py infer --config_name [config_name_without_extension]
```
