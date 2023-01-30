#Author: Nicolas DEVAUX

import torch
import torch.nn as nn
import json
#region printers (for debug)
def pd_dict_to_string(pd_dict,model)->str:
    out = f"Network: {pd_dict['model']['name']}, {pd_dict['dataset']['name']}, {pd_dict['optimizer']['name']}"
    out += f" using : {count_parameters(model)/1000000} M parameters"
    return out    
    

def dict_printer(dict_in,o:int=0):
    if len(dict_in) == 0:
        return
    sizeMax = max([len(k) for k in dict_in.keys()])
    for k,v in dict_in.items():
        if isinstance(v,dict):
            print(o*"\t" +f"{k} :")
            dict_printer(v,o+1)
        elif isinstance(v,list):
            print(o*"\t" +f"{k} :")
            list_printer(v,o+1)
        else:
            k = k + (sizeMax - len(k))*" " if len(k) < 20 else k
            print(o*"\t" +f"{k} : {v}")
    
def list_printer(list_in,o=0):
    if len(list_in) == 0:
        return
    for i,v in enumerate(list_in):
        if isinstance(v,dict):
            print(o*"\t" +f"{v} :")
            dict_printer(v,o+1)
        elif isinstance(v,list):
            print(o*"\t" +f"{v} :")
            list_printer(v,o+1)
        else:
            print(o*"\t" +f"{i} : {v}")
             
#endregion

#region Model
def get_accuracy(model:nn.Module, testloader, DEVICE):
    """Get the accuracy of a model on a testloader
    
    Args:
        model (nn.Module): The model to test
        testloader (DataLoader): The testloader to use
        DEVICE (torch.device): The device to use
    """
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

    
def count_parameters(model:nn.Module,sub:str = ""):
    """Count the number of parameters in a model
    
    Args:
        model (nn.Module): The model to count the parameters
        sub (str, optional): The name of the submodule to count the parameters. Defaults to entire model.
    """
    if sub == "":
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.get_submodule(sub).parameters())


def count_non_zero_parameters(model:nn.Module,sub:str = ""):
    """Count the number of non zero parameters in a model
    
    Args:
        model (nn.Module): The model to count the parameters
        sub (str, optional): The name of the submodule to count the parameters. Defaults to entire model.
    """
    if sub == "":
        return sum(p.to(torch.bool).sum() for p in model.parameters())
    return sum(p.to(torch.bool).sum() for p in model.get_submodule(sub).parameters())

 
#endregion