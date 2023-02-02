# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""Data Anomaly Detection
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=E0401, C0103, W0614, W0401, E0602
# noqa: E902

import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight
from utils import create_model, process_data, reduce_dimensions

if __name__ == "__main__":
    # The main function body which takens the varilable number of arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--save_model_dir',
                        default=None,
                        type=str,
                        required=False,
                        help="directory to save model to"
                        )

    parser.add_argument('-i',
                        '--intel',
                        default=0,
                        type=int,
                        required=False,
                        help="Enable this flag for intel model inference. Default is 0"
                        )

    parser.add_argument('-b',
                        '--batch_size',
                        default=128,
                        required=False,
                        type=int,
                        help="training batch size"
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
        tf.config.threading.set_inter_op_parallelism_threads(2)  # (no of sockets)
        tf.config.threading.set_intra_op_parallelism_threads(4)  # (no of physical cores)

    if FLAGS.save_model_dir is None:
        print("Please provide path to save the model...\n")
        sys.exit(1)
    else:
        save_model_path = FLAGS.save_model_dir

    model = create_model()

    x_train, y_train = process_data("./data/train.csv")

    x_reduced_inputdata, x_imp_inputdata = reduce_dimensions(
        pd.DataFrame(x_train))
    x_train = x_reduced_inputdata
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train)

    class_weights = dict(zip(np.unique(y_train), class_weights))

    stime = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=FLAGS.batch_size,
        epochs=1,
        class_weight=class_weights)
    etime = time.time()

    model.save(save_model_path + "/")

    s = f"""
        {'-' * 40}
        # Model Training
        # Time (in seconds): {etime - stime}
        # Model saved path: {save_model_path}
        {'-' * 40}
        """

    print(s)
