import torchvision
import torchvision.transforms as transforms

from typing import List

#region dataset
def get_dataset(pd_dict:dict,DATA_PATH:str):
    params = pd_dict["dataset_params"]
    
    if pd_dict["dataset"] == "CIFAR-10":
            dataset = lambda train :torchvision.datasets.CIFAR10(root=DATA_PATH +'/data', train=train,  transform=get_transformer(params["transforms"]))
            class_size = 10
    elif pd_dict["dataset"] == "CIFAR-100":
            dataset = lambda train: torchvision.datasets.CIFAR100(root=DATA_PATH +'/data', train=train,  transform=get_transformer(params["transforms"]))
            class_size = 100
    elif pd_dict["dataset"] == "SVHN":
            dataset = lambda train: torchvision.datasets.SVHN(root=DATA_PATH +'/data', split="train" if train else "test", transform=get_transformer(params["transforms"]))
            class_size = 10
    else :
            raise ValueError("Dataset not found")
    return class_size,dataset(True),dataset(False)

#endregion

#region transformers
def get_transformer(list:List[str]):
    transformList = {
        "FastAutoAugment":transforms.autoaugment.AutoAugment(), # Fix this
        "Normalize":transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "ToTensor":transforms.ToTensor()
    }
    return transforms.Compose([transformList[transform_name] for transform_name in list])

#endregion