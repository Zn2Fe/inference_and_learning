import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List,Type

def get_model(pd_dict:dict,image_size,out_size):
    params = pd_dict["model_params"]
    if pd_dict["model"] == "D-CONV":
        return D_Conv(params["alpha"],CV_Custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
    if pd_dict["model"] == "S-CONV":
        return S_Conv(params["alpha"],CV_Custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
    if pd_dict["model"] == "S-FC":
        return S_Conv(params["alpha"],FC_custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
    if pd_dict["model"] == "S-LOCAL":
        return S_Conv(params["alpha"],LL_custom,image_size,out_size,dropout_1=params["dropout_1"],dropout_2=params["dropout_2"])
    raise ValueError("Model not found")


#region DataViewLayers
class view_custom(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.view(*self.args)

class tofloat32(nn.Module):
    def __init__(self,modules:List[nn.Module]):
        super(tofloat32, self).__init__()
        self.module = nn.Sequential(*modules)
        
    def forward(self, x:torch.Tensor):
        x = x.to(torch.float32)
        x = self.module(x)
        x = x.to(torch.float16)
        return x
#endregion

#region LocalLinearImplementation
class LocalLinear_custom(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,padding : int, image_size: int):
        super(LocalLinear_custom, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        fold_num = (image_size+ 2* padding - self.kernel_size)//self.stride+1

        self.weight = nn.Parameter(torch.randn(out_channels,in_channels,fold_num,fold_num,kernel_size,kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels,fold_num,fold_num))
        
    def forward(self, x:torch.Tensor):
        x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = (x.unsqueeze(1)*self.weight).sum(dim=[-1,-2,2]) + self.bias
        return x

#endregion

#region CustomLayers (CV,LL,FC)
class interfaceModule():
    def __init__(self,size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ) -> None:
        pass
    def get_module(self) -> Tuple[int,int,nn.Module]:
        raise ValueError("interaceModule should not be called")

class CV_Custom(nn.Module,interfaceModule):
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(CV_Custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = (image_size+ 2 * padding - kernel_size)//stride+1

        self.conv = nn.Conv2d(size_in,size_out,kernel_size,stride = stride,padding = padding)
        self.batch = tofloat32([nn.BatchNorm2d(size_out,dtype = torch.float32)])
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    def get_module(self) -> Tuple[int,int,nn.Module]:
        return self.im_out,self.size_out, self

class LL_custom(nn.Module,interfaceModule):
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(LL_custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = (image_size+ 2 * padding - kernel_size)//stride+1

        self.local = LocalLinear_custom(size_in,size_out,kernel_size,stride = stride,padding = padding,image_size = image_size)
        self.batch = tofloat32([nn.BatchNorm2d(size_out,dtype = torch.float32)])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.local(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    def get_module(self) -> Tuple[int, int, nn.Module]:
        return self.im_out,self.size_out, self
    
class FC_custom(nn.Module,interfaceModule):
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(FC_custom, self).__init__()
        self.size_out = size_out
        self.im_out = image_size//stride

        self.v1 =  view_custom([-1,image_size*image_size*size_in])
        self.FC = nn.Linear(size_in * image_size**2,size_out * (image_size//stride)**2).half()
        self.v2 = view_custom([-1,size_out,image_size//stride,image_size//stride])
        self.batch = tofloat32([nn.BatchNorm2d(size_out,dtype = torch.float32)])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.v1(x)
        x = self.FC(x)
        x = self.v2(x)
        x = self.batch(x)
        x = self.relu(x)
        return x

    def get_module(self) -> Tuple[int, int, nn.Module]:
        return self.im_out,self.size_out, self
#endregion

#region CustomModel
class interfaceModel():
    def __init__(self,base_channel_size: int, net_type :Type[interfaceModule], image_size :int, out_size = 10, dropout_1:float = 0 , dropout_2:float = 0 ) -> None:
        self.conv_like : nn.Module = nn.Module()
        self.FC : nn.Module = nn.Module()
class D_Conv(nn.Module,interfaceModel):
    ### only works with square image, kernel size and stride
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
            tofloat32([nn.BatchNorm1d(24*alpha,dtype=torch.float32)]),
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


class S_Conv(nn.Module,interfaceModel):
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
            tofloat32([nn.BatchNorm1d(24*alpha,dtype=torch.float32)]),
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