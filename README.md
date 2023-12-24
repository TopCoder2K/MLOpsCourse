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

## Running Training and Evaluation

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

## Deployment with MLflow

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
{"predictions": [31.22848957148021]}
```

## Deployment with Triton

Since there are problems with the onnx version of the Random Forest model, this part is
done only for the CatBoost model.

### System configuration

The container with Triton had

```
OS:   Ubuntu 22.04.3 LTS
CPU:  12th Gen Intel(R) Core(TM) i7-12700H
vCPU: 10
RAM:  15.29GiB
```

### Deploy and test

Run the following to deploy the model:

```
docker build -t triton_with_catboost:latest mlopscourse/triton/
docker run -it --rm --cpus 10 -v ./mlopscourse/triton/model_repository:/models -v ./mlopscourse/triton/assets:/assets -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_with_catboost:latest
(You are inside the container from now)
cd mlops-course
tritonserver --model-repository /models
```

Test the model:

```
poetry run python3 mlopscourse/triton/client.py
```

The client will check the predicted output with a hardcoded value. The client should print

```
Predicted: 31.22848957148021
The test is passed!
```

### Optimization

I've used

```
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.12-py3-sdk
```

and

```
docker stats [container_id]
```

to monitor and find good configuration.

Without any optimizations:

```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 681.19 infer/sec, latency 1467 usec
Concurrency: 2, throughput: 851.402 infer/sec, latency 2348 usec
Concurrency: 3, throughput: 854.636 infer/sec, latency 3509 usec
Concurrency: 4, throughput: 821.657 infer/sec, latency 4867 usec
Concurrency: 5, throughput: 848.47 infer/sec, latency 5891 usec
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
140%.

A notable problem here is the queue. Compare for `concurrency == 1` and for
`concurrency == 5`:

```
1: Avg request latency: 1208 usec (overhead 1 usec + queue 19 usec + compute input 15 usec + compute infer 1153 usec + compute output 19 usec)
5: Avg request latency: 5631 usec (overhead 2 usec + queue 4467 usec + compute input 15 usec + compute infer 1128 usec + compute output 18 usec)
```

So, it seems I have to optimize somehow the provision of input data. Unfortunately, this
hasn't been covered in the course. Thus, let's try to optimize only the inference.

I will consider two options here: model instances and dynamic batching. Let's start with
the first one.

#### Models count

If `count: 2`, I've gotten:

```
Concurrency: 1, throughput: 574.325 infer/sec, latency 1740 usec
Concurrency: 2, throughput: 1376.2 infer/sec, latency 1452 usec
Concurrency: 3, throughput: 1608.26 infer/sec, latency 1864 usec
Concurrency: 4, throughput: 1634.76 infer/sec, latency 2446 usec
Concurrency: 5, throughput: 1642.57 infer/sec, latency 3043 usec
```

and

```
1: Avg request latency: 1425 usec (overhead 2 usec + queue 54 usec + compute input 18 usec + compute infer 1324 usec + compute output 25 usec)
5: Avg request latency: 2857 usec (overhead 2 usec + queue 1655 usec + compute input 15 usec + compute infer 1163 usec + compute output 20 usec)
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
285%. If `count: 3`, I've gotten:

```
Concurrency: 1, throughput: 491.298 infer/sec, latency 2034 usec
Concurrency: 2, throughput: 1247.34 infer/sec, latency 1602 usec
Concurrency: 3, throughput: 2019.29 infer/sec, latency 1484 usec
Concurrency: 4, throughput: 2356.69 infer/sec, latency 1696 usec
Concurrency: 5, throughput: 2385.19 infer/sec, latency 2095 usec
```

and

```
1: Avg request latency: 1640 usec (overhead 3 usec + queue 61 usec + compute input 20 usec + compute infer 1528 usec + compute output 27 usec)
5: Avg request latency: 1922 usec (overhead 2 usec + queue 682 usec + compute input 18 usec + compute infer 1199 usec + compute output 20 usec)
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
450%.

**Conclusions:**

1. The queue consumpts a lot of time compared to the model.
2. Increasing the models count from $1$ to $N$ for `concurrency == 5` results in
   $\approx 150\% \cdot N$ CPU usage, $\approx 800 \cdot N$ throughput and dividing
   queue's latency by some $k > 1$.

#### Dynamic batching

With dynamic batching `{ max_queue_delay_microseconds: 1000 }` I've gotten:

```
Concurrency: 1, throughput: 304.009 infer/sec, latency 3288 usec
Concurrency: 2, throughput: 601.324 infer/sec, latency 3324 usec
Concurrency: 3, throughput: 877.651 infer/sec, latency 3416 usec
Concurrency: 4, throughput: 1140.14 infer/sec, latency 3507 usec
Concurrency: 5, throughput: 1346.32 infer/sec, latency 3712 usec
```

and

```
1: Avg request latency: 2790 usec (overhead 3 usec + queue 1211 usec + compute input 22 usec + compute infer 1528 usec + compute output 26 usec)
5: Avg request latency: 3242 usec (overhead 5 usec + queue 1145 usec + compute input 64 usec + compute infer 1979 usec + compute output 48 usec)
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
80%. With dynamic batching `{ max_queue_delay_microseconds: 500 }` I've gotten:

```
Concurrency: 1, throughput: 409.087 infer/sec, latency 2443 usec
Concurrency: 2, throughput: 757.904 infer/sec, latency 2637 usec
Concurrency: 3, throughput: 1136.69 infer/sec, latency 2638 usec
Concurrency: 4, throughput: 1414.94 infer/sec, latency 2826 usec
Concurrency: 5, throughput: 1659.14 infer/sec, latency 3013 usec
```

and

```
1: Avg request latency: 2042 usec (overhead 2 usec + queue 671 usec + compute input 21 usec + compute infer 1323 usec + compute output 24 usec)
5: Avg request latency: 2718 usec (overhead 3 usec + queue 1196 usec + compute input 38 usec + compute infer 1444 usec + compute output 36 usec)
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
143%. With dynamic batching `{ max_queue_delay_microseconds: 100 }` I've gotten:

```
Concurrency: 1, throughput: 563.679 infer/sec, latency 1773 usec
Concurrency: 2, throughput: 852.412 infer/sec, latency 2345 usec
Concurrency: 3, throughput: 1072.07 infer/sec, latency 2797 usec
Concurrency: 4, throughput: 1429.04 infer/sec, latency 2798 usec
Concurrency: 5, throughput: 1624.53 infer/sec, latency 3076 usec
```

and

```
1: Avg request latency: 1462 usec (overhead 2 usec + queue 195 usec + compute input 19 usec + compute infer 1222 usec + compute output 22 usec)
5: Avg request latency: 2790 usec (overhead 4 usec + queue 1418 usec + compute input 31 usec + compute infer 1304 usec + compute output 33 usec)
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
147%. With dynamic batching `{ max_queue_delay_microseconds: 2000 }` I've gotten:

```
Concurrency: 1, throughput: 206.951 infer/sec, latency 4830 usec
Concurrency: 2, throughput: 419.314 infer/sec, latency 4768 usec
Concurrency: 3, throughput: 596.492 infer/sec, latency 5027 usec
Concurrency: 4, throughput: 779.399 infer/sec, latency 5130 usec
Concurrency: 5, throughput: 975.283 infer/sec, latency 5125 usec
```

and

```
1: Avg request latency: 4258 usec (overhead 3 usec + queue 2238 usec + compute input 28 usec + compute infer 1956 usec + compute output 32 usec)
5: Avg request latency: 4546 usec (overhead 5 usec + queue 2218 usec + compute input 67 usec + compute infer 2207 usec + compute output 48 usec)
```

and the approximate CPU usage by the container with the model for `concurrency == 5` is
62%.

**Conclusions:**

1. Dynamic batching can significally lower CPU usage (2 times less for `concurrency == 5`
   with `max_queue_delay_microseconds: 1000`!).
2. It seems optimal to set `max_queue_delay_microseconds: 500`, since further diminishing
   shows no improvements.

#### Best variant

Since I've allocated 10 vCPUs, let's set `count: 6` and
`{ max_queue_delay_microseconds: 500 }`:

```
Concurrency: 5, throughput: 1100.77 infer/sec, latency 4540 usec
```

and

```
5: Avg request latency: 3972 usec (overhead 7 usec + queue 676 usec + compute input 86 usec + compute infer 3137 usec + compute output 65 usec)
```

and the approximate CPU usage by the container with the model is 103%. Why the hell did
this happen? It seems that the concurrency is too small and there are no benefits from
setting `count` > 1... Let's try with `concurrency == 30`:

```
Concurrency: 30, throughput: 5393.48 infer/sec, latency 5560 usec
```

and

```
Avg request latency: 5192 usec (overhead 26 usec + queue 460 usec + compute input 320 usec + compute infer 4210 usec + compute output 175 usec)
```

and the approximate CPU usage by the container with the model is 200%. We still have a lot
of room for the vCPU usage. I've played around with `concurrency` and have found
`concurrency == 150` to be optimal:

```
Concurrency: 150, throughput: 18625.9 infer/sec, latency 8044 usec
```

and

```
Avg request latency: 7561 usec (overhead 38 usec + queue 733 usec + compute input 326 usec + compute infer 6180 usec + compute output 284 usec)
```

and the approximate CPU usage by the container with the model is 800%. With the initial
configuration I've gotten:

```
Concurrency: 150, throughput: 844.372 infer/sec, latency 176648 usec
```

and

```
Avg request latency: 176013 usec (overhead 3 usec + queue 174844 usec + compute input 18 usec + compute infer 1124 usec + compute output 23 usec)
```

and the approximate CPU usage by the container with the model is 143%.

**Conclusion:** We get 22X throughput and 0.04X latency with the best configuration for
`concurrency = 150`.

### Formal report

- [x] System configuration is at top of the section
- [x] The task description is at top of the `README.md`
- [x] My `model_repository/` is the following:
  ```
  mlopscourse/triton/model_repository/
  └── catboost
      ├── 1
      │   ├── model.py
      │   └── __pycache__
      │       ├── model.cpython-310.pyc
      │       └── model.cpython-38.pyc
      └── config.pbtxt
  ```
- [x] Experiments and motivation of the best configuration are given above
- [x] Throughput and latency metrics comparison is given above

_N.B. Since I used python backend, there is no special script for the model conversion._
