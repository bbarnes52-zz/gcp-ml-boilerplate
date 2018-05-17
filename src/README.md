# GCP ML Boilerplate

## Setup

- Install necessary dependencies

```
pip install tensorflow==1.8
pip install apache-beam==2.4.0
pip install tensorflow-transform==0.6.0
```

- Add constants to PATH

```
export PATH=${PATH}:$(pwd)
```

## Commands

- Run preprocessing locally

```
NOW=$(date +"%Y%m%d_%H%M%S")
TFT_OUTPUT_DIR=tft_outputs/${NOW}
python preprocess.py \
  --output_dir=${TFT_OUTPUT_DIR} \
  --$(gcloud config get-value project)
```

- Run training locally

```
NOW=$(date +"%Y%m%d_%H%M%S")
python ./trainer/task.py \
  --model_dir=models/${NOW} \
  --input_dir=$TFT_OUTPUT_DIR
```

- Run preprocessing on the cloud

```
BUCKET=gs://gcp-ml-boilerplate-models
TFT_OUTPUT_DIR=${BUCKET}/tft/outputs/${USER}$(date +%Y%m%d%H%M%S)
python preprocess.py \
  --output_dir $TFT_OUTPUT_DIR \
  --cloud
```
