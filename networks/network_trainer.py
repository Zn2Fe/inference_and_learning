import networks.neural_networks as nnets
import networks.custom_layers as cl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

transf = transforms.Compose([
    transforms.autoaugment.AutoAugment(), #Replace with fast autoaugment
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

line = 100
divisors = 25

class NetworkTrainer:
    modelList = {
        "D-CONV":lambda pdict,image_size,out_size: nnets.D_Conv(pdict.base_channel_size,cl.CV_Custom,image_size,out_size),
        "S-CONV":lambda pdict,image_size,out_size: nnets.S_Conv(pdict.base_channel_size,cl.CV_Custom,image_size,out_size),
        "S-FC": lambda pdict,image_size,out_size: nnets.S_Conv(pdict.base_channel_size,cl.FC_custom,image_size,out_size)
        }
    trainingMethodList = {
        "SGD":lambda params,pdict:optim.SGD(params,lr=pdict.learning_rate,momentum=pdict.momentum),
        }
    datasetList = {
    'CIFAR-10':lambda train: (10,torchvision.datasets.CIFAR10(root='./data', train=train, download=False, transform=transf)),
    "CIFAR-100":lambda train: (100,torchvision.datasets.CIFAR100(root='./data', train=train, download=False, transform=transf)),
    "SVHN":lambda train:(10,torchvision.datasets.SVHN(root='./data', split='train' if train else 'test', download=False, transform=transf))
    }
    
    def __init__(self,pdict)->None:
        #DataSets
        self.classes_size,self.trainset = self.datasetList[pdict.dataset](True)
        _,self.testset = self.datasetList[pdict.dataset](False)
        self.trainloader = DataLoader(self.trainset, batch_size=int(pdict.batch_size), shuffle=True, num_workers=2)
        self.testloader = DataLoader(self.testset, batch_size=int(pdict.batch_size), shuffle=False, num_workers=2)
        
        #Model
        self.model : nn.Module = self.modelList[pdict.model](pdict,self.trainset[0][0].shape[1],self.classes_size)
        
        #training methods
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.trainingMethodList[pdict.training_method](self.model.parameters(),pdict)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,50,T_mult=2 , last_epoch=-1)
        
        #CUDA
        self.use_cuda = pdict.use_cuda
        if self.use_cuda and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.use_cuda = False
        self.DEVICE = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(self.DEVICE)
        
        self.pdict = pdict
        self.name = pdict.model + "_" + pdict.training_method + "_" + pdict.dataset + "_" + str(pdict.batch_size) +"_"+ str(pdict.epochs)
        
    
    def train(self)->None:
        for epoch in range(self.pdict.epochs):
            for _,data in enumerate(self.trainloader,0):
                inputs, labels = data[0].to(self.DEVICE  ), data[1].to(self.DEVICE )
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            #self.scheduler.step()
            
            print('¤', end='')
            if(epoch % line == line-1): print(f" {epoch+1} : {self.getAccuracy()}%")
            elif(epoch == self.pdict.epochs-1): print(f" {epoch+1} : {self.getAccuracy()}%")
            elif(epoch % divisors == divisors-1): print('|', end='')
        
        PATH = f"networks_saved/{self.name}.pth"
        torch.save(self.model.state_dict(), PATH)
        print("Training complete, saved to :",PATH)
        self.pdict.to_json(f"networks_saved/{self.name}.json")

    
    def getAccuracy(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct // total
        

    def __str__(self)->str:
        out = f"Network Trainer: {self.pdict.model},{self.pdict.training_method},{self.pdict.dataset} -> {self.name}"
        return out 
        
    def to_string(self) -> str:
        out = "Network Trainer:\n"
        out += f"name: {self.name}\n"
        out += f"Epochs: {self.pdict.epochs}\n"
        out += f"Batch Size: {self.pdict.batch_size}\n"
        out += f"Model: {self.model.__str__()}\n"
        out += f"{self.trainset.__str__()}\n "
        out += f"Training Method: {self.optimizer.__str__()}\n"

        return out
    