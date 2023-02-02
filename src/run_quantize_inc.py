# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0115, C0116,C0415,E0401,R0914,E0611,W0108, C0103
# pylint: disable=E0401, C0103, W0614, W0401, E0602, W0621, R1722


"""
Quantize a model using Intel Neural Compressor
"""

import os
import math
import time
import argparse
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from neural_compressor.experimental import Quantization, common
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from utils import process_data, reduce_dimensions


BATCH_SIZE = 100
tf.keras.utils.set_random_seed(5)


class Dataset:

    def __init__(self, x_valid, y_valid):
        self.x_valid = x_valid
        self.y_valid = y_valid

    def __getitem__(self, index):
        return self.x_valid[index], self.y_valid[index]

    def __len__(self):
        return len(self.x_valid)

    def eval_function(self, graph_model):
        """ evaluate function to get relative accuracy of FP32
        """
        x_labels = self.y_valid
        x_test_sub = self.x_valid

        predictions = []
        total_time = 0
        batch_size = BATCH_SIZE

        concrete_function = get_concrete_function(
            graph_def=graph_model.as_graph_def()
        )

        for i in range(math.ceil(len(x_test_sub)/batch_size)):

            btch = tf.constant(
                x_test_sub[batch_size * i: batch_size * (i+1)],
                dtype=tf.float32)
            stime = time.time()
            res = concrete_function(x=btch)[0]
            total_time += (time.time() - stime)
            tmp = np.array(res)
            threshold_zero_indices = tmp < 0.5
            threshold_one_indices = tmp > 0.5
            tmp[threshold_zero_indices] = 0
            tmp[threshold_one_indices] = 1

            predictions.extend(tmp)

        print("Model inference time {} batch size {}, per sample {}".format(
            batch_size,
            (total_time/i),
            (total_time/i)/batch_size))
        print(classification_report(x_labels, predictions))
        return accuracy_score(predictions, x_labels)


class AccuracyMetric:

    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def update(self, predict, label):
        print("Len and Shape: ", len(predict), predict.shape, len(label))
        print("predict :", predict)
        print("label: ", label)
        tmp = np.array(predict)
        threshold_zero_indices = tmp < 0.5
        threshold_one_indices = tmp > 0.5
        tmp[threshold_zero_indices] = 0
        tmp[threshold_one_indices] = 1

        self.pred_list.extend(tmp)
        self.label_list.extend(label)
        self.samples += len(label)

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def result(self):
        return accuracy_score(self.pred_list, self.label_list)


def get_concrete_function(graph_def: tf.compat.v1.Graph):
    """Get a concrete function from a TF graph to
    make a callable

    Args:
        graph_def (tf.compat.v1.Graph): Graph to turn into a callable
    """

    def imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
    graph = wrap_function.graph

    return wrap_function.prune(
        tf.nest.map_structure(graph.as_graph_element, ["x:0"]),
        tf.nest.map_structure(graph.as_graph_element, ["Identity:0"]))


def quantize_model(input_graph_path, dataset, inc_config_file):
    """Quantizes the model using the given dataset and INC config

    Args:
        input_graph_path: path to .pb model.
        dataset : Dataset to use for quantization.
        inc_config_file : Path to INC config.
    """
    quantizer = Quantization(inc_config_file)
    quantizer.calib_dataloader = common.DataLoader(
        dataset, batch_size=BATCH_SIZE
    )
    quantizer.eval_dataloader = common.DataLoader(
        dataset, batch_size=BATCH_SIZE
    )
    quantizer.metric = common.Metric(AccuracyMetric)
    quantizer.eval_func = dataset.eval_function
    quantizer.model = input_graph_path
    quantized_model = quantizer.fit()

    return quantized_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--saved_frozen_graph',
        required=True,
        help="saved pretrained frozen graph to quantize",
        type=str
        )

    parser.add_argument(
        '--output_dir',
        required=True,
        help="directory to save quantized model.",
        type=str
    )

    parser.add_argument(
        '--inc_config_file',
        help="INC conf yaml",
        required=True
    )

    flags = parser.parse_args()

    if not os.path.exists(flags.saved_frozen_graph):
        print("Saved model %s not found!", flags.saved_frozen_graph)
        exit(1)

    if not os.path.exists(flags.inc_config_file):
        print("INC configuration %s not found!", flags.inc_config_file)
        exit(1)

    x_test, y_test = process_data("./data/test.csv")

    x_reduced_inputdata, x_imp_inputdata = reduce_dimensions(
        pd.DataFrame(x_test))

    x_test = x_reduced_inputdata

    print(x_test.shape)

    print("test data size: ", len(y_test))
    dataset = Dataset(x_test, list(y_test))

    quantized_model = quantize_model(
        flags.saved_frozen_graph, dataset, flags.inc_config_file)

    path = pathlib.Path(flags.output_dir)
    path.mkdir(parents=True, exist_ok=True)

    quantized_model.save(
        os.path.join(flags.output_dir, "saved_frozen_int8_model.pb")
    )
