# Inference and learning final project
Available at: https://gitlab.epfl.ch/ndevaux/inference_and_learning_final_project
## Project description
All code provided in this folder is for the final project of the course Fundamentals of Inference and Learning at EPFL. 
The goal of the project is to attempt to reproduce the following paper: https://arxiv.org/pdf/2007.13657.pdf. 

### Paper description
The paper we are trying to reproduce is focused on creating a network without inductive bias (i.e. without any prior knowledge on the data) with similar performance to convolutional network on image classification.

The paper has the main following parts:
- Isolate the effects of differents aspects of convulutional networks :
    - Local connectivity via comparing convolutional layers (CV) and fully connected layers (FC)
    - Weight sharing via comparing convolutional layers (CV) and localy connected layers (LL)
    - The effect of depth via comparing deep networks (D) and shallow networks (S)
- Propose a new optimizer algorithm called B-Lasso that is able to find FC network with similar properties as CV networks

Our goal is mainly to reproduce the second part wich are the main results of the paper. However an implementation of the network used for the first part is also provided in the library we created.
### Code short description
Our code is regrouped in a library called `networks` and two notebook called `main.ipynb` and `optimization.ipynb`. The notebook is used to run the code while the library defines functions and classes used in the notebooks.

The library has the following objectives:
- Provide a implementation of networks described in the paper
- Provide a implementation of optimizer described in the paper
- Provide an interface to describe through a json file a network and its training
- Provide a class to let the user easily train multiples networks and check different hyperparameters

The notebook `main.ipynb` has the following structure:
- Import of the library and configuration of the notebook
- A section dedicated to training the best network found in the previous section
- A section reproducing trying to reproduce the results of the paper, most notably the results of the following tables and graphs:

The notebook `optimization.ipynb` has the following structure:
- Import of the library and configuration of the notebook
- A section dedicated to optimizing the hyperparameters of the different networks encountered in the paper

## Requirements

This project is realised in python 3.8.10 and using the librairies :
- pytorch 1.13.1
- pandas 1.5.2
- numpy 1.23.4
- matplotlib 3.6.2

But may be able to run with other versions of the librairies.


The code is provided in a jupyter notebook and is compatible with google colab or other jupyter environnement.
The code will only run in a `cuda` environnement due to the use of half precision float.


## Code Description
The code is concentrated in the `networks` library, the library is organised as follow:
- `__init__.py`: provide the library with following names:
    - `Network`: class used to describe a network
    - `ThreadedTrainer`: class used to train a network
    - `optimization_thread`: function used to optimize the hyperparameters of a network
    - `n_utils`: acces point to module `utils` containing utility functions
    - `n_dest` : acces point to module `dataset` containing dataset related functions
    - `n_cl`: acces point to module `custom_layers` containing created networks and layers
    - `n_optim`: acces point to module `optimizers`  containing optimizers
- `networks.py`: contains the definition of the class `Network` used to describe a network
- `trainers.py`: contains the definition of the class `ThreadedTrainer` used to train a network
- `custom_layers.py`: contains the definition of the different layers used in the networks
- `dataset.py`: contains the definition of the dataset used in the paper
- `optimizers.py`: contains the definition of the optimizers used in the paper
- `utils.py`: contains utility functions used in the library

### `networks.py`
This module contains the definition of the class `Network` used to describe a network. The class needs to be initialised from a dictionnary, relevant information are descrived in documentation of the class.
### `trainers.py`
This module contains the definition of the class `ThreadedTrainer` used to train a network. The class works as follow:
- The class is initialised with a number of threads and PATH to relevant files
- The initialisation of the class creates a thread pool of the given size
- The class expose a queue (FIFO) of jobs to be done by the threads
- The class expose a list of results of the jobs done by the threads (and saved in the result file)
- The class expose a function to add a job to the queue
- The class expose a function `progress` to check the progress of the training
- The threads pull jobs from the queue and train corresponding networks, a snapshot is regularly saved to avoid loosing progress in case of crash
- The threads save the results of the training in the result file

With the following structure:
```
.
├── data
│   ├── dataset1
│   └── ...   
├── networks
│   └── .INCOMPLETE.json
├── non-zero.json
└── result.json
```

For example, the following code : 
- will train 3 networks in parallel
- saved the results in the file `results.json`
- saved the netwoks in the folder `networks`
- saved the number of non-zero parameters of the networks in the file `non-zero.json`

```python
from networks import ThreadedTrainer
from networks import Network
l = {...} # assuming l is a dictionnary of name,dictionnary describing networks according to the documentation of the class Network
trainer = ThreadedTrainer(
    num_threads=3, 
    DATA_PATH = "./data",
    SAVE_RESULT_PATH = "./results.json",
    SAVE_NETWORK_PATH = "./networks",
    SAVE_NON_ZERO_PATH = "./non-zero.json"
    )
for name,network in l:
    trainer.add(name,Network(network),True)

trainer.progress() # will print the progress of the training
```


### `custom_layers.py`
This module contains the definition of the different layers used in the networks. The files is organised as follow:
- `get_model`: function used to get a network from a `Network.Model` object
- DataView
    - `viewCustom` : class used to create a view of a tensor
    - `tofloat32` : a nn.Sequential layer that cast the tensor to float32 first
- LocaLinear_custom : An implementation of the LocalLinear layer
- CustomLayers : 
    - `interfaceModule` : interface for following layers to inherit from
    - `CV_Custom` : implementation of the convolution layer with Batch Norm and ReLU
    - `LL_Custom` : implementation of the localy connected layer with Batch Norm and ReLU
    - `FC_Custom` : implementation of the fully connected layer with Batch Norm and ReLU
- Custom Model
    - `D-CONV` : implementation of the D-CONV network described in the paper, this class can take as argument the custom layers to use seen previously
    - `S-CONV` : implementation of the S-CONV network described in the paper, this class can take as argument the custom layers to use seen previously
    - `FC-3` : A fully connected network with 3 layers as described in the paper
    - `ResNet18` : Although we did not use it, we implemented the ResNet18 network as described in the paper

### `dataset.py`
- `get_dataset` : function used to get a dataset from a `Network.Dataset` object
- `get_transform` : function used to get a transform from a `Network.Dataset` object

### `optimizers.py`
- `get_optimizer` : function used to get an optimizer from a `Network.Optimizer` object
- `get_scheduler` : function used to get a scheduler from a `Network.Scheduler` object

- `B_LASSO` : implementation of the B-LASSO optimizer described in the paper, this implementation is based on the implementation of the SGD optimizer from the `torch.optim` library. The class inherits from the `torch.optim.Optimizer` class and functions both in single and multi tensor mode


### `utils.py`
- printers : used for debugging purposes, print dict or list in a nice way

- `get_accuracy` : function used to get the accuracy of a network on a test Loader
- `count_parameters` : function used to get size of a network
- `count_non_zero_parameters` : function used to get number of non-zero parameters of a network



## Conclusion
Any additional information provided about the code can be found in it's documentation.

For the project, a report was written, it can be found in the `report` folder.

Please do mind that the ThreadedTrainer class has a cross reference to the ThreadedTrainer class wich I greatly regret, but I don't have the time to fix it. It may cause memory leaks if not careful.  

## Authors
- **Nicolas DEVAUX** : [nicolas.devaux@epfl.ch](mailto:nicolas.devaux@epfl.ch)
