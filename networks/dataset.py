# Author: Nicolas DEVAUX

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from typing import List,Dict,Union,Tuple,OrderedDict
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
            dataset = lambda train :torchvision.datasets.CIFAR10(root=DATA_PATH +'/data', train=train,  transform=transforms.ToTensor())
            class_size = 10
    elif des.name == "CIFAR-100":
            dataset = lambda train: torchvision.datasets.CIFAR100(root=DATA_PATH +'/data', train=train,  transform=transforms.ToTensor())
            class_size = 100
    elif des.name == "SVHN":
            dataset = lambda train: torchvision.datasets.SVHN(root=DATA_PATH +'/data', split="train" if train else "test", transform=transforms.ToTensor())
            class_size = 10
    else :
            raise ValueError(f"Dataset not found : {des.name}")
    return class_size,dataset(True),dataset(False)

#endregion

#region transformers

def get_policy(policy_name:str) -> transforms.autoaugment.AutoAugmentPolicy:
    """Gets the policy from its name"""
    if policy_name == "CIFAR-10":
        return transforms.autoaugment.AutoAugmentPolicy.CIFAR10
    elif policy_name == "CIFAR-100":
        return transforms.autoaugment.AutoAugmentPolicy.CIFAR10
    elif policy_name == "SVHN":
        return transforms.autoaugment.AutoAugmentPolicy.SVHN
    raise ValueError(f"Policy not found : {policy_name}")

def get_transform(des:Network.Dataset,DEVICE) -> torch.nn.Sequential:
    """Gets the transformer from its description

Args:
    des (dict): Description of the transformer must contain the name and the transforms(see @get_transformer)
    DEVICE (torch.device): Device to use

Returns:
    torch.nn.Sequential: The transformer
"""
    transformList:Dict[str,torch.nn.Module] = {
            
    }
    res : List[torch.nn.Module] = []
    for transform_name in des["transforms"]:
        if transform_name == "AutoAugment":
            res.append(transforms.ConvertImageDtype(torch.uint8))
            auto = transforms.autoaugment.AutoAugment(get_policy(des.name))
            auto._augmentation_space = lambda num_bins,image_size : custom_augmentation_space(num_bins,image_size,DEVICE)
            res.append(auto)
            res.append(transforms.ConvertImageDtype(torch.float16))
        elif transform_name in transformList:
            res.append(transformList[transform_name])
        else:
            raise ValueError(f"Transform not found : {transform_name}")
            
    return torch.nn.Sequential(*res).to(DEVICE)

#endregion

def custom_augmentation_space(num_bins: int, image_size: Tuple[int, int],DEVICE) -> Dict[str, Tuple[torch.Tensor, bool]]:
    """Custom augmentation space for AutoAugment cause the default one is bugged for half precision float
    This copy the default one but cast the tensors to the correct DEVICE.
    I've posted an issue on the pytorch community : 
        https://discuss.pytorch.org/t/issue-with-autoaugment-when-using-half-precision-float-torch-float16/171684
        
    """
    return {
        # op_name: (magnitudes, signed)
        "ShearX": (torch.linspace(0.0, 0.3, num_bins,device=DEVICE), True),
        "ShearY": (torch.linspace(0.0, 0.3, num_bins,device=DEVICE), True),
        "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins,device=DEVICE), True),
        "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins,device=DEVICE), True),
        "Rotate": (torch.linspace(0.0, 30.0, num_bins,device=DEVICE), True),
        "Brightness": (torch.linspace(0.0, 0.9, num_bins,device=DEVICE), True),
        "Color": (torch.linspace(0.0, 0.9, num_bins,device=DEVICE), True),
        "Contrast": (torch.linspace(0.0, 0.9, num_bins,device=DEVICE), True),
        "Sharpness": (torch.linspace(0.0, 0.9, num_bins,device=DEVICE), True),
        "Posterize": (8 - (torch.arange(num_bins,device=DEVICE) / ((num_bins - 1) / 4)).round().int(), False),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins,device=DEVICE), False),
        "AutoContrast": (torch.tensor(0.0,device=DEVICE), False),
        "Equalize": (torch.tensor(0.0,device=DEVICE), False),
        "Invert": (torch.tensor(0.0,device=DEVICE), False),
    }