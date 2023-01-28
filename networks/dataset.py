# Author: Nicolas DEVAUX

import torchvision
import torchvision.transforms as transforms

from typing import List

#region dataset


def get_dataset(des:dict,DATA_PATH:str):
    
    if des["name"] == "CIFAR-10":
            dataset = lambda train :torchvision.datasets.CIFAR10(root=DATA_PATH +'/data', train=train,  transform=get_transformer(des["transforms"]))
            class_size = 10
    elif des["name"] == "CIFAR-100":
            dataset = lambda train: torchvision.datasets.CIFAR100(root=DATA_PATH +'/data', train=train,  transform=get_transformer(des["transforms"]))
            class_size = 100
    elif des["name"] == "SVHN":
            dataset = lambda train: torchvision.datasets.SVHN(root=DATA_PATH +'/data', split="train" if train else "test", transform=get_transformer(des["transforms"]))
            class_size = 10
    else :
            raise ValueError(f"Dataset not found : {des['name']}")
    return class_size,dataset(True),dataset(False)

#endregion

#region transformers

def get_transformer(list:List[str]):
    transformList = {
        "ToTensor":transforms.ToTensor(),
        "FastAutoAugment":transforms.autoaugment.AutoAugment(), # Fix this
        "Normalize":transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    }
    return transforms.Compose([transformList[transform_name] for transform_name in list])

#endregion