# Author: Nicolas DEVAUX

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from typing import List,Dict,Union,Tuple
from networks.network import Network
#region dataset


def get_dataset(des:Network.Dataset,DATA_PATH:str) -> Tuple[int,Dataset,Dataset]:
    """Gets the dataset from its description

Args:
    des (dict): Description of the dataset must contain the name and the transforms(see @get_transformer)
    DATA_PATH (str): Path to the data folder

Raises:
    ValueError: If the dataset is not found

Returns:
    Tuple[int,Dataset,Dataset]: The class size, the training dataset and the test dataset
    """    
    if des.name == "CIFAR-10":
            dataset = lambda train :torchvision.datasets.CIFAR10(root=DATA_PATH +'/data', train=train,  transform=get_transformer(des["transforms"]))
            class_size = 10
    elif des.name == "CIFAR-100":
            dataset = lambda train: torchvision.datasets.CIFAR100(root=DATA_PATH +'/data', train=train,  transform=get_transformer(des["transforms"]))
            class_size = 100
    elif des.name == "SVHN":
            dataset = lambda train: torchvision.datasets.SVHN(root=DATA_PATH +'/data', split="train" if train else "test", transform=get_transformer(des["transforms"]))
            class_size = 10
    else :
            raise ValueError(f"Dataset not found : {des.name}")
    return class_size,dataset(True),dataset(False)

#endregion

#region transformers

def get_transformer(list:List[str]) -> transforms.Compose:
    """Gets the transformer from its description

Args:
    list (List[str]): List of the transforms to use

Returns:
    transforms.Compose: The composed transformer
"""
    transformList:Dict[str,Union[transforms.ToTensor,torch.nn.Module]] = {
        "ToTensor":transforms.ToTensor(),
        "FastAutoAugment":transforms.autoaugment.AutoAugment(), # Fix this
        "Normalize":transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    }
    return transforms.Compose([transformList[transform_name] for transform_name in list])

#endregion