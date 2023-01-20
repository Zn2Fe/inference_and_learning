import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalLinear_custom(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,padding : int, image_size: int):
        super(LocalLinear_custom, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fold_num = (image_size+ 2* padding - self.kernel_size)//self.stride+1
        
        in_features = self.in_channels*self.fold_num**2
        out_features = self.out_channels*self.fold_num**2
        self.conv = torch.nn.Conv2d(in_features, out_features, self.kernel_size, stride=1, padding=0)
        
    def forward(self, x:torch.Tensor):
        x = F.pad(x, [self.padding]*4, value= 0)
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = x.flatten(1,3)
        x = self.conv(x)
        x = x.view(-1, self.out_channels, self.fold_num, self.fold_num)
        return x

class view_custom(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.view(*self.args)

class interfaceModule():
    def __init__(self,size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ) -> None:
        pass
    def get_module(self) -> tuple[int,int,nn.Module]:
        print("interfaceModule should not be called")
        return 0,0,nn.Module()

class CV_Custom(nn.Module,interfaceModule):
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(CV_Custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = (image_size+ 2 * padding - kernel_size)//stride+1

        self.conv = nn.Conv2d(size_in,size_out,kernel_size,stride = stride,padding = padding)
        self.batch = nn.BatchNorm2d(size_out)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    def get_module(self) -> tuple[int,int,nn.Module]:
        return self.im_out,self.size_out, self


class LL_custom(nn.Module,interfaceModule):
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(LL_custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = (image_size+ 2 * padding - kernel_size)//stride+1

        self.local = LocalLinear_custom(size_in,size_out,kernel_size,stride = stride,padding = padding,image_size = image_size)
        self.batch = nn.BatchNorm2d(size_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.local(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    def get_module(self) -> tuple[int, int, nn.Module]:
        return self.im_out,self.size_out, self
    
class FC_custom(nn.Module,interfaceModule):
    def __init__(self, size_in: int, size_out: int, kernel_size: int, stride: int, image_size: int ):
        super(FC_custom, self).__init__()
        padding = kernel_size//2
        self.size_out = size_out
        self.im_out = image_size//stride

        self.v1 =  view_custom([-1,image_size*image_size*size_in])
        self.FC = nn.Linear(size_in * image_size**2,size_out * (image_size//stride)**2)
        self.v2 = view_custom([-1,size_out,image_size//stride,image_size//stride])
        self.batch = nn.BatchNorm2d(size_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.v1(x)
        x = self.FC(x)
        x = self.v2(x)
        x = self.batch(x)
        x = self.relu(x)
        return x

    def get_module(self) -> tuple[int, int, nn.Module]:
        return self.im_out,self.size_out, self
