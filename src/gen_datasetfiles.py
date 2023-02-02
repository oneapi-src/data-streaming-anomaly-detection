# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""Data Anomaly Detection
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=E0401

import argparse
import pandas as pd


def process_data(data_path):
    """Perform the preprocessing of the data set

    Args:
        data_path: The path of the input data file

    Returns:
        None
    """
    print ("Generation of the files train.csv and test.csv in progress!")
    print ("Please wait...")
    dataframe = pd.read_csv(data_path, encoding="latin-1")
    without_anomaly = dataframe[dataframe.anomaly == 0.0]
    with_anomaly = dataframe[dataframe.anomaly == 1.0]

    without_anomaly_row_length = len(without_anomaly)
    with_anomaly_row_length = len(with_anomaly)
    nr_train_samples_without_anomaly = int(without_anomaly_row_length * 0.8)
    nr_train_samples_with_anomaly = int(with_anomaly_row_length * 0.8)

    df_new = without_anomaly[0:nr_train_samples_without_anomaly]
    df_new = df_new.append(with_anomaly[:nr_train_samples_with_anomaly], ignore_index=True)

    df_new.to_csv("train.csv", index=False)

    df_new = without_anomaly[nr_train_samples_without_anomaly:without_anomaly_row_length]
    df_new = df_new.append(with_anomaly[nr_train_samples_with_anomaly:with_anomaly_row_length], ignore_index=True)

    df_new.to_csv("test.csv", index=False)
    print ("Generation of files completed.")

if __name__ == "__main__":
    # The main function body which takens the varilable number of arguments

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--dataset_file',
                        '--dataset_file',
                        type=str,
                        required=True,
                        default=None,
                        help='dataset file for training')

    # Holds all the arguments passed to the function
    FLAGS = PARSER.parse_args()

    process_data(FLAGS.dataset_file)
