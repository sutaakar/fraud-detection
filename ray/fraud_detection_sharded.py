import os

import ray
import tensorflow as tf

from ray import train
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer

import pandas as pd

from ray.data.preprocessors import Concatenator
from ray.data.preprocessors import StandardScaler

from ray.train.tensorflow.keras import ReportCheckpointCallback


import s3fs
import pyarrow

import numpy as np

import tf2onnx
import onnx


s3_bucket = os.getenv('AWS_S3_BUCKET')

def get_s3_filesystem():
    return s3fs.S3FileSystem(
        key=os.getenv('AWS_ACCESS_KEY_ID'),
        secret=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_S3_ENDPOINT'),
        client_kwargs={
           'verify': False
        }
    )


def get_minio_run_config():
    storage_filesystem = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(get_s3_filesystem()))
    run_config = ray.train.RunConfig(storage_path=s3_bucket, storage_filesystem=storage_filesystem)
    return run_config


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    return model


def train_func(config: dict):
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 3)
    train_columns = config.get("train_columns")

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_model(train_columns)
        multi_worker_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    dataset = train.get_dataset_shard("train")

    tf_dataset = dataset.to_tf(
        feature_columns="distance_from_last_transaction", label_columns="fraud", batch_size=batch_size
    )

    results = []
    for epoch in range(epochs):
        history = multi_worker_model.fit(tf_dataset, callbacks=[ReportCheckpointCallback()])
        results.append(history.history)
    return results


url = "https://raw.githubusercontent.com/rh-aiservices-bu/fraud-detection/main/data/card_transdata.csv"
Data = pd.read_csv(url)
Data = Data.drop(columns = ['repeat_retailer','distance_from_home'])

Data.head()
print(Data.head())

train_dataset = ray.data.from_pandas(Data)

preprocessor = StandardScaler(columns=["distance_from_last_transaction", "ratio_to_median_purchase_price"])
scaled_train_dataset = preprocessor.fit_transform(train_dataset)

concatenator = Concatenator(output_column_name="distance_from_last_transaction", exclude=["fraud"])
concatenated_train_dataset = concatenator.fit_transform(scaled_train_dataset)

number_of_train_columns = len(train_dataset.columns())-1
config = {"batch_size": 128, "epochs": 1, "train_columns": number_of_train_columns}

scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
trainer = TensorflowTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=scaling_config,
    datasets={"train": concatenated_train_dataset},
    run_config=get_minio_run_config()
)
result = trainer.fit()


checkpoint = result.checkpoint
with checkpoint.as_directory() as local_checkpoint_dir:
    trained_model = tf.keras.models.load_model(local_checkpoint_dir + '/model.keras')


threshold = 0.95

sally_ds = ray.data.from_items(
    [{"distance_from_last_transaction": 0.3111400080477545, "ratio_to_median_purchase_price": 1.9459399775518593, "used_chip": 1.0, "used_pin_number": 0.0, "online_order": 0.0}]
)
sally_values = preprocessor.transform(sally_ds).to_pandas().values
print(sally_values)

prediction = trained_model.predict(sally_values)
print(prediction)

print("Is Sally's transaction predicted to be fraudulent? (true = YES, false = NO) ")
print(np.squeeze(prediction) > threshold)

print("How likely was Sally's transaction to be fraudulent? ")
print("{:.5f}".format(np.squeeze(prediction) * 100) + "%")


fraud_ds = ray.data.from_items(
    [{"distance_from_last_transaction": 1.55200812594914, "ratio_to_median_purchase_price": 4.60360068820619, "used_chip": 1.0, "used_pin_number": 0.0, "online_order": 1.0}]
)
fraud_values = preprocessor.transform(fraud_ds).to_pandas().values
print(fraud_values)

prediction = trained_model.predict(fraud_values)
print(prediction)

print("Is fraud transaction predicted to be fraudulent? (true = YES, false = NO) ")
print(np.squeeze(prediction) > threshold)

print("How likely was fraud transaction to be fraudulent? ")
print("{:.5f}".format(np.squeeze(prediction) * 100) + "%")

model_proto, _ = tf2onnx.convert.from_keras(trained_model)
os.makedirs("models/fraud/1", exist_ok=True)
onnx.save(model_proto, "models/fraud/1/model.onnx")

with open("models/fraud/1/model.onnx", 'rb') as source_file:
    with get_s3_filesystem().open(f"{s3_bucket}/models/fraud/1/model.onnx", 'wb') as destination_file:
        destination_file.write(source_file.read())
