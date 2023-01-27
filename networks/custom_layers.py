# Author: Nicolas DEVAUX

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List,Type,OrderedDict,overload

def get_model(des:dict,image_size,out_size):
    if des["name"] in ["D-CONV","D-FC","D-LOCAL","S-CONV","S-FC","S-LOCAL"]:
        lm,cm = des["name"].split("-")
        cl = {"CONV":CV_Custom,"FC":FC_custom,"LOCAL":LL_custom}[cm]
        ll = {"D":D_Conv,"S":S_Conv}[lm]
        return ll(des["alpha"],cl,image_size,out_size,dropout_1=des["dropout_1"],dropout_2=des["dropout_2"])
    if des["name"] == "3-FC":
        return FC_3(des["alpha"],image_size,out_size)
    raise ValueError("Model not found")


#region DataViewLayers

class view_custom(nn.Module):
    """Custom view layer
    """    
    def __init__(self,args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.view(*self.args)

class tofloat32(nn.Sequential):
    """A sequential layer that convert the input to float32, apply the layers and convert back to float16
    """    
    def __init__(self, *args) -> None:
        super(tofloat32,self).__init__(*args)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = super().forward(x)
        x = x.to(torch.float16)
        return x
         
#endregion

#region LocalLinearImplementation


# Implementation inspired by https://github.com/pytorch/pytorch/issues/499#issuecomment-503962218
# and unfold documentation https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
class LocalLinear_custom(nn.Module):
    """Custom Local Linear layer, python implementation, heavy memory usage"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,padding : int, image_size: int):
        super(LocalLinear_custom, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fold_num = ((image_size+ 2* padding - self.kernel_size)//self.stride+1)
        self.im = self.fold_num**2
        self.col = self.kernel_size**2 * self.in_channels

        self.weight = nn.Parameter(torch.randn(out_channels,self.fold_num,self.fold_num,in_channels,kernel_size,kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels,self.fold_num,self.fold_num))
        
    def forward(self, x:torch.Tensor):
        x = torch.nn.functional.unfold(x,(self.kernel_size,self.kernel_size), padding= self.padding, stride= self.stride)
        x = x.view(x.size(0),1,self.im,self.col) * self.weight.view(self.out_channels,self.im,self.col)
        x = x.sum(dim=-1) + self.bias.view(self.out_channels,self.im)
        return x.view(x.size(0),self.out_channels,self.fold_num,self.fold_num)

#endregion

#region CustomLayers (CV,LL,FC)
class interfaceModule():
    """Interface for custom layers"""
    def __init__(self,size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ) -> None:
        pass
    def get_module(self) -> Tuple[int,int,nn.Module]:
        raise ValueError("interaceModule should not be called")

class CV_Custom(nn.Module,interfaceModule):
    """Custom Convolutional layer, apply batch norm and relu
    
    Args:
        size_in (int): number of input channels
        size_out (int): number of output channels
        kernel_size (int): size of the kernel
        stride (int): stride of the convolution
        image_size (int): size of the image (not used for this layer)
    """
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(CV_Custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = (image_size+ 2 * padding - kernel_size)//stride+1

        self.conv = nn.Conv2d(size_in,size_out,kernel_size,stride = stride,padding = padding)
        self.batch = tofloat32(nn.BatchNorm2d(size_out,dtype = torch.float32))
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    def get_module(self) -> Tuple[int,int,nn.Module]:
        """Return the size of the output and the module itself"""
        return self.im_out,self.size_out, self

class LL_custom(nn.Module,interfaceModule):
    """Custom Local Linear layer, apply batch norm and relu
    
    Args:
        size_in (int): number of input channels
        size_out (int): number of output channels
        kernel_size (int): size of the kernel
        stride (int): stride of the convolution
        image_size (int): size of the image 
    """
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(LL_custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = (image_size+ 2 * padding - kernel_size)//stride+1

        self.local = LocalLinear_custom(size_in,size_out,kernel_size,stride = stride,padding = padding,image_size = image_size)
        self.batch = tofloat32(nn.BatchNorm2d(size_out,dtype = torch.float32))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.local(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    def get_module(self) -> Tuple[int, int, nn.Module]:
        """Return the size of the output and the module itself"""
        return self.im_out,self.size_out, self
    
class FC_custom(nn.Module,interfaceModule):
    """Custom Fully Connected layer, apply batch norm and relu
    
    Args:
        size_in (int): number of input channels
        size_out (int): number of output channels
        kernel_size (int): size of the kernel (not used for this layer)
        stride (int): stride (only reduced the image output size)
        image_size (int): size of the image
    """
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(FC_custom, self).__init__()
        self.size_out = size_out
        self.im_out = image_size//stride

        self.v1 =  view_custom([-1,image_size*image_size*size_in])
        self.FC = nn.Linear(size_in * image_size**2,size_out * (image_size//stride)**2).half()
        self.batch = tofloat32(nn.BatchNorm1d(size_out * (image_size//stride)**2,dtype = torch.float32))
        self.v2 = view_custom([-1,size_out,image_size//stride,image_size//stride])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.v1(x)
        x = self.FC(x)
        x = self.batch(x)
        x = self.v2(x)
        x = self.relu(x)
        return x

    def get_module(self) -> Tuple[int, int, nn.Module]:
        return self.im_out,self.size_out, self
#endregion

#region CustomModel
#region S_CONV and D_CONV
class interfaceModel():
    def __init__(self,base_channel_size: int, net_type :Type[interfaceModule], image_size :int, out_size = 10, dropout_1:float = 0 , dropout_2:float = 0 ) -> None:
        self.conv_like : nn.Module = nn.Module()
        self.FC : nn.Module = nn.Module()

# implementation of the Dense Convolutional Network according to :
# https://arxiv.org/abs/2007.13657
class D_Conv(nn.Module,interfaceModel):
    """Dense Convolutional Network with custom layers
    
    Args:
        base_channel_size (int): base channel size (alpha in the paper)
        net_type (Type[interfaceModule]): type of the custom layer
        image_size (int): size of the image
        out_size (int, optional): number of output classes. Defaults to 10.
        dropout_1 (float, optional): dropout rate for the first dropout layer. Defaults to 0.
        dropout_2 (float, optional): dropout rate for the second dropout layer. Defaults to 0.
    """
    def __init__(        self,
        base_channel_size: int,
        net_type :Type[interfaceModule],
        image_size :int,
        out_size = 10,
        dropout_1 = 0,
        dropout_2 = 0 ) -> None:
        super(D_Conv).__init__()
        
        alpha = base_channel_size
        channel_size :int = 3
        im_size : int = image_size
        
        modules : List[nn.Module] = []
        size_conv = [
            (alpha,1),
            (2*alpha,2),
            (2*alpha,1),
            (4*alpha,2),
            (4*alpha,1),
            (8*alpha,2), 
            (8*alpha,1),
            (16*alpha,2)] 
        
        for i,val in enumerate(size_conv):
            size_out, stride = val
            im_size,channel_size,new_module = net_type(channel_size,size_out,3,stride,im_size).get_module()
            modules.append(new_module)
            
        self.conv_like = nn.Sequential(*modules)
        
        modules = [
            nn.Linear(channel_size*im_size**2,24*alpha).half(),
            tofloat32(nn.BatchNorm1d(24*alpha,dtype=torch.float32)),
            nn.ReLU()
        ]
        if dropout_1 != 0:
            modules.append(nn.Dropout(dropout_1))
        modules.append(nn.Linear(24*alpha,out_size).half())
        if dropout_2!=0:
            modules.append(nn.Dropout(dropout_2))
        self.FC = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_like(x)
        x = torch.flatten(x,1)
        x = self.FC(x)
        return x


# implementation of the Shallow Convolutional Network according to :
# https://arxiv.org/abs/2007.13657
class S_Conv(nn.Module,interfaceModel):
    """Shallow Convolutional Network with custom layers
    
    Args:
        base_channel_size (int): base channel size (alpha in the paper)
        net_type (Type[interfaceModule]): type of the custom layer
        image_size (int): size of the image
        out_size (int, optional): number of output classes. Defaults to 10.
        dropout_1 (float, optional): dropout rate for the first dropout layer. Defaults to 0.
        dropout_2 (float, optional): dropout rate for the second dropout layer. Defaults to 0.
    """
    def __init__(
        self,
        base_channel_size: int,
        net_type :Type[interfaceModule],
        image_size :int,
        out_size = 10,
        dropout_1 = True,
        dropout_2 = True) -> None:
        super().__init__()
        
        alpha = base_channel_size
        channel_size :int = 3
        im_size : int = image_size
        
        modules : List[nn.Module] = []
        
        im_size,channel_size, self.conv_like = net_type(channel_size,alpha,9,2,im_size).get_module()
        
        modules = [
            nn.Linear(channel_size*im_size**2,24*alpha).half(),
            tofloat32(nn.BatchNorm1d(24*alpha,dtype=torch.float32)),
            nn.ReLU()
        ]
        if dropout_1 != 0:
            modules.append(nn.Dropout(dropout_1))
        modules.append(nn.Linear(24*alpha,out_size).half())
        if dropout_2!=0:
            modules.append(nn.Dropout(dropout_2))
        self.FC = nn.Sequential(*modules)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_like(x)
        x = torch.flatten(x,1)
        x = self.FC(x)
        return x
#endregion

# implementation of a Fully Connected Network according to :
# https://arxiv.org/abs/2007.13657
class FC_3(nn.Module):
    """Fully Connected Network with 2 hidden layers
    
    Args:
        alpha (int): base channel size (size of hidden layers)
        image_size (int): size of the image
        out_size (int, optional): number of output classes. Defaults to 10.
        
    """
    def __init__(
        self,
        alpha :int,
        image_size :int,
        out_size = 10
        ) -> None:
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(image_size*image_size*3,alpha).half(),
            tofloat32(nn.BatchNorm1d(alpha,dtype=torch.float32)),
            nn.ReLU(),
            nn.Linear(alpha,alpha).half(),
            tofloat32(nn.BatchNorm1d(alpha,dtype=torch.float32)),
            nn.ReLU(),
            nn.Linear(alpha,out_size).half()
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x,1)
        x = self.FC(x)
        return x    

# region ResNet
# implementation of the ResNet18 according to : https://arxiv.org/abs/1512.03385
# implementation inspired by : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


class BasicBlock(nn.Module):
    """Basic Block for ResNet"""
    def __init__(self,in_size,out_size,stride,downsample) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        if downsample :
            self.downsample = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size),
            )
        else:
            self.downsample = None
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return x
class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1, downsample=False),
            BasicBlock(64, 64, stride=1, downsample=False),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True),
            BasicBlock(128, 128, stride=1, downsample=False),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True),
            BasicBlock(256, 256, stride=1, downsample=False),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True),
            BasicBlock(512, 512, stride=1, downsample=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# endregion

#endregion