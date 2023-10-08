# MLOpsCourse

## Task Description

The "Bike Rentals" dataset is used for scripts in this repository. This dataset contains
daily counts of rented bicycles from the bicycle rental company
[Capital-Bikeshare](https://capitalbikeshare.com/) in Washington D.C., along with weather
and seasonal information. The goal is to predict how many bikes will be rented depending
on the weather and the day. The dataset info is

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

To setup only the necessary dependencies, run the following:

```
poetry install --without dev
```

If you want to use `pre-commit`, install all the dependencies:

```
poetry install
```

## Running experiments

### Training

If you want to train the chosen model and save it afterwards, run:

```
poetry run python3 commands.py train --model_type [chosen_model]
```

The available models are Random Forest (from the scikit-learn library) and CatBoost.

### Evaluation

If you want to infer a previously trained model, make sure you've placed the checkpoint in
`checkpoints/` and then run

```
poetry run python3 commands.py infer --model_type [chosen_model] --ckpt [checkpoint_filename_with_extension]
```
