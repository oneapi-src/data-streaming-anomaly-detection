# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Time Series Synthetic Data Generation
"""

# pylint: disable=R0913, W0613, C0415,C0116, E0401,R0914,E1129,E0611
# pylint: disable=C0103, W0614, W0401, E0602
# pylint: disable=R1705, C0303, W0201, R0902, W0621, E0401
# noqa: E502

import random
import argparse
import numpy as np
import pandas as pd


class synTimeSeriesGen:
    """
    Base synthetic timeseries data generation class
    """

    def __init__(
            self,
            start_time="2021-01-01 00:00:00",
            end_time="2021-01-02 00:00:00",
            time_interval=1,
    ):
        """
        Initialize

        Args:
            start_time (str): Begin time in the format 'YYYY-MM-DD hh:mm:ss'
            end_time (str): End time in the format 'YYYY-MM-DD hh:mm:ss'
            time_interval (float): Time (in minutes) for the unit process

        Returns: None
        """
        self.start_time = start_time
        self.end_time = end_time
        self.process_time = time_interval
        etime = np.datetime64(self.end_time)
        stime = np.datetime64(self.start_time)
        self._duration_ = etime - stime
        self._minutes_ = self._duration_ / np.timedelta64(1, "m")
        self._size_ = int(self._minutes_ / (self.process_time))
        self.time_arr = np.arange(
            self.start_time,
            self.end_time,
            step=self.process_time,
            dtype="datetime64[m]",
        )

        self._normal_flag_ = False
        self._anomaly_flag_ = False
        self._drifted_flag_ = False

        self.anomaly_list = []

    def __repr__(self):
        return "Customized synthetic time series data generation class"

    def normal_distribution(
            self,
            loc=0.0,
            scale=1.0,
            return_df=True,
    ):
        """
        Initiates the normal process data

        Args:
            loc (float): Parameter (mean) for the Gaussian distribution
            scale (float): Parameter (std.dev) for the Gaussian distribution

        Returns:
            DataFrame: If `return_df=True` returns dataframe with two columns -
            `time` with the datetime values,
            `normal_data` with the normal process data
            numpy.ndarray: If `return_df=False` returns just the array of
            normal process data
        """
        self.loc = loc
        self.scale = scale
        arr = np.random.normal(
            loc=self.loc,
            scale=self.scale,
            size=self._size_)
        self.normal_data = arr
        self._normal_flag_ = True
        if return_df:
            df = pd.DataFrame(
                {
                    "time": self.time_arr, "normal_data": self.normal_data
                    })
            return df
        else:
            return self.normal_data

    def gen_timeseries_anomaly(
                        self,
                        size=1000,
                        anomaly_fraction=0.02,
                        anomaly_scale=2.0,
                        loc=0.0,
                        scale=1.0):
        """
        Generates a time-series data (array) with some anomalies

        Arguments:
            size: Size of the array
            anomaly_fraction: Fraction anomalies
            anomaly_scale: Scale factor of anomalies
            loc: Parameter (mean) for the underlying Gaussian distribution
            scale: Parameter (std.dev) for the underlying Gaussian distribution
        """
        np.random.seed()

        arr = np.random.normal(loc=loc, scale=scale, size=self._size_)

        arr_min = arr.min()
        arr_max = arr.max()
        no_anomalies = int(self._size_*anomaly_fraction)

        np.random.seed(49)

        idx_list = np.random.choice(a=size, size=no_anomalies, replace=False)
        self.anomaly_list = np.sort(idx_list)

        for idx in idx_list:
            low = arr_min-anomaly_scale*(arr_max-arr_min)
            if not max(low, 0):
                low = 0

            high_value = random.randint(int(0.5 * arr_max), int(arr_max))

            arr[idx] = loc+np.random.uniform(
                                low=low,
                                high=arr_max + high_value)

        np.random.seed()
        arrnoise = np.random.normal(loc=2, scale=5, size=self._size_)

        arr = arrnoise + arr
        return arr

    def gen_timeseries_dataframe(
                        self, n=30, prob_anomolous=0.1,
                        size=1000, anomaly_fraction=0.02, anomaly_scale=2.0,
                        loc=0.0, scale=1.0):
        """
        Generates dataframe of time-series containing 'normal'
            and 'anomolous' samples

        Arguments:
            n: Number of time-series
            prob_anomolous: Probability a time-series containing anomalies
            size: Size of the array (individual time-series)
            anomaly_fraction: Fraction anomalies
            anomaly_scale: Scale factor of anomalies
            loc: Parameter (mean) for the underlying Gaussian distribution
            scale: Parameter (std.dev) for the underlying Gaussian distribution

        Returns:
            A dataframe of shape (n,2) where the first column contains
            time-series data as list and the second column contains the binary
            classification of 0 (normal) or 1 (anomolous)
        """
        if prob_anomolous < 1.0:
            print("Prob anomaly greater than 1.0")

        dt = {}
        for i in range(n):
            anomolous = np.random.uniform(0, 1)
            k = random.randint(0, n)

            if anomolous < prob_anomolous:
                dt[str(i)] = [
                    self.gen_timeseries_anomaly(
                        size=self._size_,
                        anomaly_fraction=anomaly_fraction,
                        anomaly_scale=anomaly_scale*k/2,
                        loc=loc*k, scale=scale),
                    1]
            else:
                dt[str(i)] = [self.gen_timeseries_anomaly(
                    size=self._size_,
                    anomaly_fraction=0.0,
                    anomaly_scale=anomaly_scale*k/2,
                    loc=loc*k, scale=scale), 0]
            print("completed generation of feature {} ".format(i))

        df = pd.DataFrame(dt).T
        df.columns = ['ts', 'anomolous']
        return df

    def set_normal_distribution_data(self, time=None, data=None):
        if time is not None:
            self.time_arr = time

        if data is not None:
            self.normal_data = data

    def get_anomaly_values(self):
        sublist = [self.normal_data[i] for i in self.anomaly_list]
        sub_timearr = [self.time_arr[i] for i in self.anomaly_list]

        return sublist, sub_timearr

    def get_time_arr(self):
        return self.time_arr

    def get_anomaly_list(self):
        return self.anomaly_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-st',
                        '--start_time',
                        type=str,
                        required=True,
                        default=None,
                        help='time in the format \'YYYY-MM-DD hh:mm:ss\'')

    parser.add_argument('-et',
                        '--end_time',
                        type=str,
                        required=True,
                        default=None,
                        help='end time in the format \'YYYY-MM-DD hh:mm:ss\'')

    parser.add_argument('-ti',
                        '--time_interval',
                        type=int,
                        required=False,
                        default=1,
                        help='Time (in minutes) for the unit process')

    parser.add_argument('-nf',
                        '--number_of_features',
                        type=int,
                        required=False,
                        default=10,
                        help='Number of features to generate')

    parser.add_argument('-m',
                        '--mean',
                        type=int,
                        required=False,
                        default=10,
                        help='mean value for normal distribution')

    parser.add_argument('-af',
                        '--anomaly_fraction',
                        type=int,
                        required=False,
                        default=5,
                        help='%% of anomaly to be induced into generated data,\
                            Ex: 5 means 5%% of anomalies will be induced.')

    parser.add_argument('-f',
                        '--file_name',
                        type=str,
                        required=True,
                        default=None,
                        help='name of the file to save the generated data')

    FLAGS = parser.parse_args()
    start_time = FLAGS.start_time
    end_time = FLAGS.end_time
    time_interval = FLAGS.time_interval
    number_of_features = FLAGS.number_of_features
    loc = FLAGS.mean
    anomaly_fraction = FLAGS.anomaly_fraction
    filename = FLAGS.file_name

    ts = synTimeSeriesGen(start_time, end_time, time_interval)

    tsDataFrame = ts.gen_timeseries_dataframe(
        n=number_of_features,
        loc=loc,
        scale=5,
        prob_anomolous=0.9,
        anomaly_fraction=(anomaly_fraction/100))

    anomaly_list = ts.get_anomaly_list()
    timearr = ts.get_time_arr()

    output_values = np.zeros(len(timearr))
    for index in range(len(timearr)):
        if index in anomaly_list:
            output_values[index] = 1

    features_data = []
    features_data.append(timearr)
    features_names = ["timestamp"]
    for index in range(number_of_features):
        features_data.append(tsDataFrame.ts[index])
        features_names.append("sensor"+str(index))

    features_data.append(output_values)

    newdf = pd.DataFrame(list(zip(*features_data)))

    features_names.append("anomaly")
    newdf.columns = features_names

    newdf.to_csv("data/"+filename, header=True, index=False)
