
<a name="readme-top"></a>

<h3 align="center">RLDec</h3>

  <p align="center">
    This project is an implementation of the ATLAS decomposition approach as described in the thesis "Application of Machine Learning for Refactoring Legacy Systems in Vehicle Production" <a href="https://arxiv.org/">(2025)</a>.

  </p>

# Installation

1. Clone the repo
   ```sh
   git clone https://github.com/
   ```

2. Install the Agent ATLAS as a Python module:
   ```sh
   cd src/decomp_agent_atlas
   pip install -e .
   ```

# Usage

In order to perfrom a decomposition on one of the existing open-source projects plants, jpetstore, daytrader or roller follow the following steps. You can decide whether you want to perform a training with specific hyperparameters or perform a hyperparamter optimization using optuna search.

To perform training with hyperparameter tuning:

1. Add and remove the datasets you want to decompose in the ```experiments/standard_decomposition.py``` to the ```dataset_names``` and ```dataset_paths``` lists
2. In ```standard_decomposition.py``` set:
  ```python
   perform_hyperparameter_tuning = True
   ```
3. Configure the search spaces for the hyperparameters in ```configs/config_provider.py```
4. Static hyperparameters that are not part of the search space can be customized in the corresponding config file of the data set in the ```/configs/[dataset_name]``` directory
5. Set the number of trials you want to perform per dataset for hyperparameter tuning in the ```standard_decomposition.py``` file by changing the variable ```n_trials```
6. Execute the script ```standard_decomposition.py``` 

The results can be found in the corresponding output directory under ```data/decompositions/new_decompositions```

To perform one training / decomposition with your config:

1. Configure the hyperparameters in the corresponding config file of the data set in the ```/configs/[dataset_name]``` directory
2. In ```standard_decomposition.py``` set:
  ```python
   perform_hyperparameter_tuning = False
   ```
3. Add and remove the datasets you want to decompose in the ```experiments/standard_decomposition.py``` to the ```dataset_names``` and ```dataset_paths``` lists
4. Execute the script ```standard_decomposition.py``` 

The results can be found in the corresponding output directory under ```data/decompositions/new_decompositions```

In order for ALTLAS to decompose your call graph, you need to analyze and parse your applications source code using a dedicated parser. The agent only requires a single adjacency matrix for a standard decomposition. For db focused decomposition please refer to the adjacency matrixes that werde discussed in the thesis. When you have obtained the adjacency matrix of you application follow the next steps:

1. Config your agent setup and hyperparameters in a new config file in the config directory
2. Add your config class in the ```config_provider.py``` analogly to the other datasets
3. Set up the dataset_name, dataset_path in ```standard_decomposition.py``` 
4. Run ```standard_decomposition.py``` with your setup
5. The decomposition results can be found in the ```data/decompositions/new_decompositions/[your_dataset_name]``` directory

# References
The code in this project was implemented with the help of the Claude Sonnet 4.5 by Anthropic. 