# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""Data Anomaly Detection
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=E0401, C0103, W0614, W0401, E0602

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


RANDOMSTATE = 101


def create_model():
    """Create the model architecture

    Args:
        None
    Returns:
        model: The compiled model
    """
    tf.random.set_seed(42)

    initializer = tf.keras.initializers.HeNormal()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, axis=-1),
        input_shape=[None]))
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                kernel_initializer=initializer,
                return_sequences=False)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=sgd,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    model.build()
    model.summary()

    return model


def process_data(fname):
    """Read the csv file and pre-process the data

    Args:
        fname: name of the csv file
    Returns:
        x_data: Independent features of the dataset
        y_data: Dependent feature of the dataset
    """
    df = pd.read_csv(fname)

    df = df.reset_index()

    # The first column has no name. Give it a name!
    df.rename(columns={df.columns[0]: "Samplenr"}, inplace=True)

    fill_value = -1.0
    data_with_gaps_filled = df.fillna(fill_value)

    data_with_gaps_filled.isna().sum()

    column_length = len(data_with_gaps_filled.columns)
    row_length = len(data_with_gaps_filled)

    # Drop the first two columns and use only the sensor values
    sensordata_cols_only = data_with_gaps_filled.iloc[:, 2:column_length]
    # create a MinMaxScaler with feature range [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_sensor_data = scaler.fit_transform(sensordata_cols_only)

    input_vec_len = sensordata_cols_only.shape[1]

    nr_train_samples = row_length
    x_data = scaled_sensor_data[0:nr_train_samples, 0:input_vec_len]
    y_data = df.iloc[0:nr_train_samples, -1]

    return x_data, y_data


def reduce_dimensions(inputdata):
    """Read the csv file and pre-process the data

    Args:
        inputdata: The dataset to perform the feature extraction
    Returns:
        reduced_inputdata: The dataset with the reduced features
        most_important_sensors: The important features selected by PCA
    """

    # "Performs dimensionality reduction using PCA"
    pca = PCA(n_components=.99, svd_solver='full', random_state=RANDOMSTATE)
    pca.fit(inputdata)
    print(f'Number of components after reduction: {pca.n_components_}')

    n_comps = pca.n_components_

    most_important_comps = \
        [np.abs(pca.components_[i]).argmax() for i in range(n_comps)]
    intital_features = list(inputdata.columns)

    most_important_sensors = \
        [intital_features[most_important_comps[i]] for i in range(n_comps)]

    reduced_inputdata = pca.transform(inputdata)
    print("PCA transformed  shape:", reduced_inputdata.shape)

    return reduced_inputdata, most_important_sensors
