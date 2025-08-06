## OCDetect  - Dataset

This repository contains the code for the creation of our OCDetect dataset, as well as for the replication of its
validation results.

### Requirements:

- The downloaded dataset: (url goes here TODO)
___
#### Personal "skills":
In order to reproduce the results of our dataset creation and evaluation you should feel confident that you 
are able to do the following:
- Installing Python and creating a virtual environment
- Using a command line to run a python script
- Editing a configuration file to include your local file paths
___
#### Hard and Software:
- For the execution of DL models, an NVIDIA-GPU may be required. The code will be slow without one.
- The project uses python 3.10.

Instructions to be followed before running the experiments:
- A new conda or virtual environment
- install the required python packages: `pip install -r requirements.txt`
- run the experiments using `./run_main.sh` or `python main.py`, but first edit the configuration (see below)



Configuration:
The file `misc/config/config.yaml` must be adapted to include your environment.
```
- configuration-name:
      username: your-local-user
      hostname: your-local-hostname
      output_folder: ../OCDetect_results/Analysis/plots/idle_regions/
      export_subfolder: ../OCDetect_results/preprocessed/
      export_subfolder_ml_prepared: ../OCDetect/
      plots_folder: ../OCDetect/Analysis/plots/
      ml_results_folder: ../OCDetect/Analysis/ml_results/
      prefix: OCDetect_
```

___
      
### Structure of the Code:

The code is structured in three functional parts:
1. Dataset cleansing, for the creation of the dataset from its raw data. The dataset is cleaned from empty recordings,
regions in recordings without any activities and obviously incorrect labels
2. Preprocessing: Reading in of the finalized dataset, filtering, feature creation and scaling as well as creation of sliding windows
3. Machine Learning: Code to search for an optimal classical machine learning model


#### Running (parts of) the code
- Depending on the selections in the config, different parts will be executed 
- As mentioned above, use `python main.py` to run the experiments

#### Results

The following results can be generated and placed into the results folder:
- Statistics:
- ML Results

___

### Credits
#### Using our code
Our code can be used for research purposes. If you use parts of it or our dataset please cite the paper:

