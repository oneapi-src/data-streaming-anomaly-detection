# Dataset preparation steps

The benchmarking scripts expects the synthetic data to be generated before execution of the scripts.
The synthetic data is generated using random normal distribution with different means. Each distribution with specific mean can be 
treated as an output from a sensor connected to machine equipment. Then the anomalies are added using a uniform distribution as 
per specified percentage of anomalies required. Then a normal distribution with small mean is added as noise data.

Use the `synDataGeneration.py` script with appropriate parameters to generate synthetic data for training and inference after 
activating a conda environment. Stock environment is used as reference below.

The below commands can be used to automate the creation of a conda environment for execution of the algorithms.

> Note: It is assumed that the present **working directory is the root directory of this code repository**

```sh
conda env create --file env/stock/anomalydetection_stock.yml
```
This command utilizes the dependencies found in the `env/stock/anomalydetection_stock.yml` file to create an environment as follows:

YAML file: `env/stock/anomalydetection_stock.yml`

Environment Name: `anomalydetection_stock`

Configuration: **Python=3.9.7 with TensorFlow 2.8.0**

Use the following command to activate the environment that was created and run the script:

```sh
conda activate anomalydetection_stock
python synDataGeneration.py
```

The script `synDataGeneration.py` takes the below arguments.

```
usage: src/synDataGeneration.py [-h] -st START_TIME -et END_TIME [-ti TIME_INTERVAL] [-nf NUMBER_OF_FEATURES] [-m MEAN] [-af ANOMALY_FRACTION] -f FILE_NAME

optional arguments:
  -h, --help            show this help message and exit
  -st START_TIME, --start_time START_TIME
                        start time in the format 'YYYY-MM-DD hh:mm:ss'
  -et END_TIME, --end_time END_TIME
                        end time in the format 'YYYY-MM-DD hh:mm:ss'
  -ti TIME_INTERVAL, --time_interval TIME_INTERVAL
                        Time (in minutes) for the unit process
  -nf NUMBER_OF_FEATURES, --number_of_features NUMBER_OF_FEATURES
                        Number of features to generate
  -m MEAN, --mean MEAN  mean value for normal distribution
  -af ANOMALY_FRACTION, --anomaly_fraction ANOMALY_FRACTION
                        % of anomaly to be induced into generated data, Ex: 5 means 5% of anomalies will be induced.
  -f FILE_NAME, --file_name FILE_NAME
                        name of the file to save the generated data
```

As an example of using this to generate synthetic data for below parameters is given below:

```
start_time = ""2019-10-01 00:00:00"
end_time = "2020-10-30 00:00:00"
time_interval = 1 (min)
number_of_features = 150
mean = 12
anomaly_fraction = 1%
file_name = "synData.csv"
```

```sh
conda activate anomalydetection_stock
python src/synDataGeneration.py -st "2019-10-01 00:00:00" -et "2020-10-30 00:00:00" -ti 1 -nf 150 -m 12 -af 1 -f "synData.csv"
```

which will produce a synthetic data in `data/synData.csv` file which can be used for running training and inference.

Now with these parameters as an example, the script will generate samples for each minute between the time interval "2019-10-01 00:00:00" to "2020-10-30 00:00:00". 
So, for each day there will be 1440 samples generated. For the given parameters, a total of approximately 1400*365 samples will be generated.
The number of features to be generated is given as 150, hence each feature will have approximately 1400*365 samples.
For each feature, a normal distribution data is generated with the given mean of 12 and random standard distribution within a specific range.
Next using the anomaly fraction value given as parameter, which is equal to 1 in the above command, the number of anomalies to be generated is determined.

> Number of anomalies = Total Number of Samples * (1/100) = 1440*365 * (0.01)

So total number of anomalies will be approximately 1440*365 * (0.01) samples.

Next using a uniform distribution for the required number of anomalies, the anomalies are introduced on the above generated normal distribution for each feature. 
Then a normal distribution with mean (0<= mean<=2) and standard deviation of 1 is generated and added on top the final distribution to create noise data. 
With these steps, the final distribution of time series data with 1% anomalies is generated and can be used as a dataset for our benchmark analysis.

The generated dataset file, under `data` folder, is used for benchmarking the inference time of the model to show performance gain using Intel® optimization for TensorFlow 2.11.0 over the stock version of TensorFlow 2.8.0.
