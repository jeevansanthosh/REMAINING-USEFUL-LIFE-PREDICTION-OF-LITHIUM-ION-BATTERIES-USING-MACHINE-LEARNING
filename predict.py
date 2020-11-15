"""
Given an input, output a prediction using a saved model
"""

import os
import pickle
import argparse
import datetime
import logging
import pandas as pd
import tensorflow as tf
from utils import preprocessing
from build_dataset import import_csv
from build_dataset import drop_columns


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    default=None,
)
parser.add_argument(
    "--model_dir",
    default=None,
    help="Path directory containing the model"
)
# REVIEW: maybe hard code the name ?
parser.add_argument(
    "--dataset_name",
    default=None,
    help="Name of the dataset"
)


if __name__ == '__main__':
    # Parser setup
    args = parser.parse_args()
    # Disable logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Args tests
    assert os.path.isfile(args.data_file), "No data file found"
    assert os.path.isdir(args.model_dir), "No model directory found"

    # Import data
    data = import_csv(args.data_file)
    data = drop_columns(data)

    # Preprocessing
    data = preprocessing(data, inference=True, name=args.dataset_name)

    # Load the model
    model = tf.keras.models.load_model(args.model_dir)

    # Inference time !
    # We multiply by 60 to convert minutes (prediction) to seconds
    prediction = int(model.predict(data.values)[0][0].round()*60)
    output = str(datetime.timedelta(seconds=prediction))
    print("Predicted battery life : {}".format(output))
