# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415, E0401,R0914,E1129,E0611
# pylint: disable=E0401, C0103, W0614, W0401, E0602

# noqa: E502

"""
Run inference with benchmarks on Tensorflow native models.

"""

import math
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from utils import create_model, process_data, reduce_dimensions

tf.keras.utils.set_random_seed(42)


def load_pb(in_model: str) -> tf.compat.v1.Graph:
    """Load a frozen graph from a .pb file

    Args:
        in_model (str): .pb file

    Returns:
        tf.compat.v1.Graph: tensorflow graph version
    """
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(in_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')
    return detection_graph


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--saved_frozen_model',
                        default=None,
                        type=str,
                        required=False,
                        help="saved frozen graph."
                        )

    parser.add_argument('-i',
                        '--intel',
                        default=0,
                        type=int,
                        required=False,
                        help="Enable this flag for intel model inference. Default is 0"
                        )

    parser.add_argument('-d',
                        '--duration',
                        default=30,
                        type=int,
                        required=False,
                        help="Duration in Minutes"
                        )

    parser.add_argument(
        '--benchmark_mode',
        type=bool,
        default=False,
        help="benchmark inference time with respect to 'num_iters', default will be on entire test dataset "
    )

    parser.add_argument('-n',
                        '--num_iters',
                        default=100,
                        type=int,
                        required=False,
                        help="number of iterations to use when benchmarking"
                        )

    parser.add_argument('-bf16',
                        '--bf16',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to enable bf16 capablities, \
                                default is 0')

    FLAGS = parser.parse_args()

    if FLAGS.bf16 == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision_mkl": True})
        print("Mixed Precision Enabled!")

    if FLAGS.intel == 1:
        tf.config.threading.set_inter_op_parallelism_threads(2) # (no of sockets)
        tf.config.threading.set_intra_op_parallelism_threads(4) # (no of physical cores)

    model = create_model()

    x_test, y_test = process_data("./data/test.csv")

    x_reduced_inputdata, x_imp_inputdata = reduce_dimensions(
        pd.DataFrame(x_test))

    x_test = x_reduced_inputdata

    print(x_test.shape)

    # load model which is saved as a frozen graph
    model = load_pb(FLAGS.saved_frozen_model)
    concrete_function = get_concrete_function(
        graph_def=model.as_graph_def()
    )

    x_labels = y_test
    x_test_sub = x_test
    if FLAGS.benchmark_mode:
        times = []
        predictions = []
        xlabels = []
        for i in range(10+FLAGS.num_iters):
            idx = np.random.randint(x_test_sub.shape[0], size=FLAGS.duration)
            btch = tf.constant(x_test_sub[idx], dtype=tf.float32)
            start = time.time()
            res = concrete_function(x=btch)
            end = time.time()
            if i > 10:
                times.append(end - start)

            tmp = np.array(res)
            threshold_zero_indices = tmp < 0.5
            threshold_one_indices = tmp > 0.5
            tmp[threshold_zero_indices] = 0
            tmp[threshold_one_indices] = 1

            tmp = tmp.flatten()
            predictions.extend(tmp)
            tlabels = x_labels[idx]
            tlabels = tlabels.reset_index(drop=True)
            xlabels.extend(tlabels)

        print(classification_report(predictions, xlabels))

        s = f"""
        {'-'*40}
        # Model Inference details:
        # Average inference:
        #   Time (in seconds): {np.mean(times)}
        #   Sensor Data Duration : {FLAGS.duration}
        {'-'*40}
        """
        print(s)
    else:
        predictions = []
        total_time = 0
        for i in range(math.ceil(len(x_test_sub)/FLAGS.duration)):
            btch = tf.constant(
                x_test_sub[FLAGS.duration * i: FLAGS.duration * (i+1)],
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

        print(classification_report(x_labels, predictions))

        s = f"""
        {'-'*40}
        # Model Inference details:
        # Average inference:
        #   Time (in seconds): {(total_time/i)}
        #   Sensor Data Duration : {FLAGS.duration}
        {'-'*40}
        """
        print(s)
