# MLOpsCourse

## Setup

To setup only the necessary dependencies, run the following:

```
poetry install --without dev
```

If you want to use `pre-commit`, install all the dependencies:

```
poetry install
```

## Run experiments

To train and evaluate the chosen model, run:

```
poetry run python3 main.py --model [chosen_model]
```

Note that the `--model` argument is optional. By default, the scripts use Random Forest.

If you only want to train the chosen model and save it afterwards, run:

```
poetry run python3 mlopscourse/train.py --model [chosen_model]
```

If you only want to infer a previously trained model, make sure you've placed the
checkpoint in `checkpoints/` and then run

```
poetry run python3 mlopscourse/infer.py --model [chosen_model] --ckpt [checkpoint_name]
```

The `--ckpt` argument is also optional. The script uses `model_[model_name_from_args].p`
as the default filename. _But if you set `--model`, do not forget to set `--ckpt` also!_
