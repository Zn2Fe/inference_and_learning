

class Network():
    def __init__(self,network:dict) -> None:
        """A class encapsulating all the information needed to create a network

        Args:
            network (dict): must contain the following keys:
                - batch_size (int): batch size to use
                - epoch (int): number of epoch to train
                - use_cuda (bool): whether to use cuda or not
                - optimizer (dict): Network.Optimizer
                - model (dict): see Network.Model
                - dataset (dict): see Network.Dataset
                - loss (dict): see Network.Scheduler
        """
        self.batch_size:int = network["batch_size"]
        self.epoch :int = network["epoch"]
        self.use_cuda :str = network["use_cuda"]
        self.optimizer = Network.Optimizer(network["optimizer"])
        self.model = Network.Model(network["model"])
        self.dataset = Network.Dataset(network["dataset"])
        self.scheduler = Network.Scheduler(network["scheduler"])
    class Optimizer():
        def __init__(self,optimizer) -> None:
            """A class encapsulating all the information needed to create an optimizer

            Args:
                optimizer (dict): must contain the following keys:
                    - name (str): name of the optimizer
                    - lr (float): learning rate
                    - ...
                    
            """
            self.name = optimizer["name"]
            if not "lr" in optimizer:
                raise ValueError("learning rate must be specified in optimizer")
            self.params = {k:v for k,v in optimizer.items() if k != "name"}
        def __getitem__(self,key):
            return self.params[key]
        
    class Model():
        def __init__(self,model:dict) -> None:
            """"A class encapsulating all the information needed to create a model
            
            Args:
                model (dict): must contain the following keys:
                    - name (str): name of the model
            """
            self.name = model["name"]
            self.params = {k:v for k,v in model.items() if k != "name"}
        def __getitem__(self,key):
            return self.params[key]
    class Dataset():
        def __init__(self,dataset:dict) -> None:
            """A class encapsulating all the information needed to create a dataset

            Args:
                dataset (dict): must contain the following keys:
                    - name (str): name of the dataset
                    - transforms (list): list of transforms to apply to the dataset
            """
            self.name = dataset["name"]
            self.params = {k:v for k,v in dataset.items() if k != "name"}
        def __getitem__(self,key):
            return self.params[key]
    class Scheduler():
        def __init__(self,scheduler:dict) -> None:
            """A class encapsulating all the information needed to create a scheduler

            Args:
                scheduler (dict): must contain the following keys:
                    - name (str): name of the scheduler
            """
            self.name = scheduler["name"]
            self.params = {k:v for k,v in scheduler.items() if k != "name"}
        def __getitem__(self,key):
            return self.params[key]