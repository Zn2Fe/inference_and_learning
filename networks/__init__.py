
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from typing import Tuple

import networks.utils as n_utils
from networks.custom_layers import get_model
from networks.dataset import get_dataset
from networks.optimizers import get_optimizer,get_scheduler

progress = lambda model, testLoader,DEVICE: print("%.2f" % n_utils.get_accuracy(model, testLoader,DEVICE),end='%, ')
line = (10,lambda epoch,epoch_max: print(f' : {epoch}/{epoch_max}'))
div = (25, lambda epoch,epoch_max: print('',end=""))




def NetworkTrainer(pd_dict:dict,verbose =False, very_verbose = False,DATA_PATH:str = ".",saveTo=None)->Tuple[float,nn.Module]:
    #CUDA
    use_cuda = pd_dict["use_cuda"]
    if use_cuda and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        use_cuda = False
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    
    #DataSets
    class_size,trainset,testset = get_dataset(pd_dict,DATA_PATH)
    dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=pd_dict["batch_size"], shuffle= train, num_workers=2)
    trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
    
    #Model
    model :nn.Module = get_model(pd_dict,trainset[0][0].shape[1],class_size).to(DEVICE)
    
    #Training method
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(pd_dict,model)
    scheduler = get_scheduler(pd_dict,optimizer)
    
    #Training
    if verbose: print(f"Training {n_utils.pd_dict_to_string(pd_dict,model)} on {DEVICE}")
    if very_verbose: n_utils.dict_printer(pd_dict,0)
    
    for epoch in range(pd_dict["epoch"]):
        for _,(inputs,labels) in enumerate(trainLoader,0):
            inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if verbose:
            progress(model, testLoader,DEVICE)
            if(epoch % line[0] == line[0]-1): line[1](epoch+1,pd_dict["epoch"])
            elif(epoch == pd_dict["epoch"]-1): line[1](epoch+1,pd_dict["epoch"])
            elif(epoch % div[0] == div[0]-1): div[1](epoch+1,pd_dict["epoch"])
    if verbose:
        print("Finish training !")
        print(f"Accuracy : {n_utils.get_accuracy(model, testLoader,DEVICE)}%")
    
    if saveTo != None :
      torch.save(model,saveTo)
    
    return n_utils.get_accuracy(model, testLoader,DEVICE),model.to("cpu")


def NetworkOptimizer(pd_dict:dict,verbose = True, very_verbose = False,DATA_PATH:str=".") -> Tuple[dict,bool]:
        optimize = pd_dict["optimize"]
        if len(optimize) == 0:
            return pd_dict,False
        best = pd_dict
        
        best["epoch"] = optimize["epoch"]
        changed = False

        if verbose : print("Calculating base accuracy...")
        best["accuracy"],_ =  NetworkTrainer(best,verbose = very_verbose,very_verbose = False,DATA_PATH=DATA_PATH)
        if verbose : print(f"Got accuracy : {best['accuracy']} for base")
        for key,value in optimize.items():
            if key == "epoch":
                continue
            if verbose: print(f"Optimizing {key} with values {value}")
            res = {key: value, "accuracy": []}
            key_1,key_2 = key.split(".")
            
            for v in value:
                if v == best[key_1][key_2]:
                    if verbose : print(f"Got accuracy : {best['accuracy']} for {key} = {v}")
                    res["accuracy"].append(best["accuracy"])
                    continue
                buff = best[key_1][key_2]
                best[key_1][key_2] = v
                acc,_ = NetworkTrainer(best,verbose = very_verbose,DATA_PATH=DATA_PATH)
                res["accuracy"].append(acc)
                if verbose : print(f"Got accuracy : {acc} for {key} = {v}") 
                if  acc > best["accuracy"]:
                    best["accuracy"] = acc
                    changed = True
                else :
                  best[key_1][key_2] = buff

            if verbose : print(pd.DataFrame(res))
            if verbose : print(f"Best value for {key} is {best[key_1][key_2]} with accuracy {best['accuracy']}")
        return best,changed

 