import networks.neural_networks as nnets
import networks.custom_layers as cl
from networks.printer import pd_dict_to_string,dict_printer
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List
import pandas as pd


line = 100
divisors = 25

def get_model(pd_dict:dict,image_size,out_size):
    params = pd_dict["model_params"]
    match pd_dict["model"]:
        case "D-CONV":
            return nnets.D_Conv(params["alpha"],cl.CV_Custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
        case "S-CONV":
            return nnets.S_Conv(params["alpha"],cl.CV_Custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
        case "S-FC":
            return nnets.S_Conv(params["alpha"],cl.FC_custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
    raise ValueError("Model not found")

def get_optimizer(pd_dict:dict):
    tr_params = pd_dict["optimizer_params"]
    match pd_dict["optimizer"]:
        case "SGD":
            return lambda params:optim.SGD(params,lr=tr_params["lr"],momentum=tr_params["momentum"],weight_decay=tr_params["weight_decay"])
    raise ValueError("Training method not found")

def get_dataset(pd_dict:dict):
    params = pd_dict["dataset_params"]
    
    match pd_dict["dataset"]:
        case "CIFAR-10":
            dataset = lambda train :torchvision.datasets.CIFAR10(root='./data', train=train,  transform=get_transformer(params["transforms"]))
            class_size = 10
        case "CIFAR-100":
            dataset = lambda train: torchvision.datasets.CIFAR100(root='./data', train=train,  transform=get_transformer(params["transforms"]))
            class_size = 100
        case "SVHN":
            dataset = lambda train: torchvision.datasets.SVHN(root='./data', split="train" if train else "test", transform=get_transformer(params["transforms"]))
            class_size = 10
        case _:
            raise ValueError("Dataset not found")
    return class_size,dataset(True),dataset(False)

def get_transformer(list:List[str]):
    transformList = {
        "FastAutoAugment":transforms.autoaugment.AutoAugment(), # Fix this
        "Normalize":transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "ToTensor":transforms.ToTensor(),
    }
    return transforms.Compose([transformList[transform_name] for transform_name in list])

def get_scheduler(pd_dict:dict):
    params = pd_dict["scheduler_params"]
    match pd_dict["scheduler"]:
        case "CosineAnnealingWarmRestarts":
            return lambda optimizer :optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = params["T_0"],
                T_mult=params["T_mult"],
                last_epoch=params["last_epoch"])
    raise ValueError("Scheduler not found")

def get_accuracy(model, testloader,DEVICE):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        model.to(DEVICE)
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total   


def NetworkTrainer(pd_dict:dict,verbose =False, very_verbose = False)->tuple[float,nn.Module]:
    #CUDA
    use_cuda = pd_dict["use_cuda"]
    if use_cuda and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        use_cuda = False
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    
    #DataSets
    class_size,trainset,testset = get_dataset(pd_dict)
    dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=pd_dict["batch_size"], shuffle= train, num_workers=2)
    trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
    
    #Model
    model :nn.Module = get_model(pd_dict,trainset[0][0].shape[1],class_size).to(DEVICE)
    
    #Training method
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(pd_dict)(model.parameters())
    scheduler = get_scheduler(pd_dict)(optimizer)
    
    #Training
    if verbose: print(f"Training {pd_dict_to_string(pd_dict,model)} on {DEVICE}")
    if very_verbose: dict_printer(pd_dict,0)
    
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
            print('¤', end='')
            if(epoch % line == line-1): print(f" {epoch+1}: {get_accuracy(model, testLoader,DEVICE)} %")
            elif(epoch == epoch-1): print(f" {epoch+1}")
            elif(epoch % divisors == divisors-1): print('|', end='')
    if verbose:
        print("Finish training !")
        print(f"Accuracy : {get_accuracy(model, testLoader,DEVICE)}%")
    return get_accuracy(model, testLoader,DEVICE),model.to("cpu")




def NetworkOptimizer(pd_dict:dict,optimize:dict,verbose = True, very_verbose = False) -> dict:
        best = pd_dict
        if verbose : print("Calculating base accuracy")
        best["accuracy"],_ =  NetworkTrainer(best,verbose = verbose,very_verbose = very_verbose)
        for key,value in optimize.items():
            if verbose: print(f"Optimizing {key} with values {value}")
            res = {key: value, "accuracy": []}
            key_1,key_2 = key.split(".")
            for v in value:
                if v == best[key_1][key_2]:
                    res["accuracy"].append(best["accuracy"])
                    continue
                copy = best.copy()
                copy[key_1][key_2] = v
                acc,net = NetworkTrainer(copy,verbose = verbose,very_verbose = very_verbose)
                res["accuracy"].append(acc) 
                if  acc > best["accuracy"]:
                    best["accuracy"] = acc
                    best[key_1][key_2] = v
            if verbose : print(pd.DataFrame(res))
            if verbose : print(f"Best value for {key} is {best[key_1][key_2]} with accuracy {best['accuracy']}")
        return best
        
def networktable(name:str,path:str,train=False,optimize= False,verbose = True, very_verbose = False):
    database = pd.read_json(path+"/"+name+".json")
    for i in range(database.__len__()):
        data = database.iloc[i].to_dict()
        if train:
            database.iloc[i]["accuracy"],net = NetworkTrainer(data,verbose = verbose,very_verbose = very_verbose)
            torch.save(net,path+"/"+name+f"_{i}.pth")

 
