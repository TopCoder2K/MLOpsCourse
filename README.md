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

The command should download two .csv files from my
[GDrive](https://drive.google.com/drive/folders/1fCTKCtocuLIhDQ5OaL8lQKtI8fPcBVFZ?usp=sharing)
and place them inside the `mlopscourse/data/` directory.

## Running experiments

### Training

If you want to train the chosen model and save it afterwards, place its configuration file
in the `configs` directory and run:

```
poetry run python3 commands.py train --config_name [config_name_without_extension]
```

The available models are `rf` (Random Forest from the `scikit-learn` library) and `cb`
(Yandex's CatBoost), so an example with the CatBoost would be the following:

```
poetry run python3 commands.py train --config_name cb_config
```

_N.B. Do not forget to set `logging.mlflow.tracking_uri` before the launch. The logs are
saved in the default directory: `mlruns`. If you are using the standard MLFlow server,
then run it before the training with `poetry run mlflow ui`._

### Evaluation

If you want to infer a previously trained model, make sure you've placed the checkpoint in
`checkpoints/` and the configuration file in `configs/` then run

```
poetry run python3 commands.py infer --config_name [config_name_without_extension]
```

### Deployment with MLflow

**Warning! This feature works stably only with the CatBoost model.** Predictions of the
onnx version of the Random Forest differ from the original one (see
[this](https://github.com/onnx/sklearn-onnx/issues/1047#issuecomment-1851837537)).
Moreover, I was not able to infer the onnx version with MLflow (although everything worked
fine with the `mlflow.sklearn` flavour as you can see in the `hw2` version of the
repository).

In order to deploy a trained model, run:

```
poetry run mlflow models serve -p 5001 -m checkpoints/mlflow_[model_type]_ckpt/ --env-manager=local
```

where `[model_type]` is `cb` or `rf`.

After this, it is possible to send requests to the model. I've created a script to
generate the correct json containing the first example from the training set, but the json
itself is in the repository, so you can skip this step. If you want to generate the json
by yourself, run:

```
poetry run python3 create_example_request.py create_example_request
```

Send a request to the deployed model using the generated json:

```
curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d @example_request.json
```

The model should reply with something like this:

```
{"predictions": [20.8]}
```
