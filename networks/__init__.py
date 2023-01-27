# Author: Nicolas DEVAUX

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

import networks.utils as n_utils
import networks.dataset as dset
import networks.custom_layers as cl
import networks.optimizers as noptim




class NetworkOptim:
    """Interface to optimize the hyperparameters of a network
    
    Args:
        optim (dict): dictionary containing the hyperparameters to optimize
        DATA_PATH (str): path to the loacation of the dataset
        epoch (int, optional): number of epoch to train the network. Defaults to 100.
        verbose (bool, optional): verbose mode. Defaults to False.
        very_verbose (bool, optional): very verbose mode. Defaults to False.
        
    """
    def __init__(self,optim:dict,DATA_PATH,epoch=100,verbose = False, very_verbose = False) -> None:
        self.optim = optim
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.DATA_PATH = DATA_PATH
        self.epoch = epoch
    
    def __call__(self,net:dict):
        """Optimize the hyperparameters of a network
        
        Args:
            net (dict): dictionary containing the network to optimize
        """
        net["epoch"] = self.epoch
        if self.verbose: print("Calculating accuracy for default parameters")
        accuracy,_,_ = train(net,self.DATA_PATH,verbose=self.very_verbose)
        best = {}
        for to_optim in ["model","dataset","optimizer","scheduler"]:
            for key,item in self.optim[to_optim][net[to_optim]["name"]].items():
                if self.verbose:print(f"Optimizing {to_optim}.{key} with {item}")
                best[to_optim+"."+key] = net[to_optim][key]
                acc,item = self.__train_list(net,to_optim,key,[i for i in item if i!= net[to_optim][key]])
                if acc > accuracy:
                    accuracy = acc
                    best[to_optim+"."+key] = item
                net[to_optim][key] = best[to_optim+"."+key]
                if self.verbose:print(f"Best accuracy for {to_optim}.{key} : {accuracy} with {best[to_optim+'.'+key]}")
        return accuracy,best
                
    def __train_list(self,net,key1,key2,items):
        accuracy = 0
        best = items[0]
        for item in items:
            net[key1][key2] = item
            if self.verbose: print(f"Calculating accuracy for {key1}.{key2} = {item}")
            acc,_,_ = train(net,self.DATA_PATH,verbose=self.very_verbose)
            if acc > accuracy:
                accuracy = acc
                best = item
            if self.verbose: print(f"Accuracy for {key1}.{key2} = {item} : {acc}")
        return accuracy,best

# implementation according to pytorch documentation
def train(net:dict, DATA_PATH,save_number_of_non_zero = False,verbose=False):
    """Train a network
    
        Args:
            net (dict): dictionary containing the network to train
            DATA_PATH (str): path to the location of the dataset
            save_number_of_non_zero (bool, optional): save the number of non zero parameters for S_[CONV,LOCAL,FC]. Defaults to False.
            verbose (bool, optional): verbose mode. Defaults to False.
        
        Returns:
            accuracy (float): accuracy of the network
            model (nn.Module): trained model
            number_of_non_zero (dict): number of non zero parameters for S_[CONV,LOCAL,FC]
    """ 
    n_utils.dict_printer(net)
    #CUDA
    use_cuda = net["use_cuda"]
    if use_cuda and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        use_cuda = False
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    
    #DataSets
    class_size,trainset,testset = dset.get_dataset(net["dataset"],DATA_PATH)
    dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=net["batch_size"], shuffle= train, num_workers=2)
    trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
    
    #Model
    model :nn.Module = cl.get_model(net["model"],trainset[0][0].shape[1],class_size).to(DEVICE)
    number_of_non_zero = {}
    if save_number_of_non_zero:
        if not isinstance(model,cl.S_Conv):
            raise ValueError("Model must be S_Conv")
        number_of_non_zero = {
            "conv_like":[n_utils.get_number_of_non_zero_parameters(model,"conv_like")],
            "FC1":[n_utils.get_number_of_non_zero_parameters(model,"FC.0")],
            "FC2":[n_utils.get_number_of_non_zero_parameters(model,"FC.3")],
        }
    #Training method
    criterion = nn.CrossEntropyLoss()
    optimizer = noptim.get_optimizer(net["optimizer"],model)
    scheduler = noptim.get_scheduler(net["scheduler"],optimizer)
    
    if verbose: print(f"Training {n_utils.pd_dict_to_string(net,model)} on {DEVICE}")
    for epoch in range(net["epoch"]):
        for _,(inputs,labels) in enumerate(trainLoader,0):
            inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if verbose:
            print("%.2f" % n_utils.get_accuracy(model, testLoader,DEVICE),end='%, ')
            if(epoch % 10 == 10-1) : print(f' : {epoch+1}/{net["epoch"]}')
            elif(epoch == net["epoch"]-1): print(f' : {epoch+1}/{net["epoch"]}')
        if save_number_of_non_zero:
            number_of_non_zero["conv_like"].append(n_utils.get_number_of_non_zero_parameters(model,"conv_like"))
            number_of_non_zero["FC1"].append(n_utils.get_number_of_non_zero_parameters(model,"FC.0"))
            number_of_non_zero["FC2"].append(n_utils.get_number_of_non_zero_parameters(model,"FC.3"))
    
    acc = n_utils.get_accuracy(model, testLoader,DEVICE)
    if verbose:
        print("Finish training !")
        print(f"Accuracy : {acc}%")
    return acc,model,number_of_non_zero