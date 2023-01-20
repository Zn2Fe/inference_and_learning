import torch
import torch.nn as nn
import networks.custom_layers as cl

from typing import Type,List

class D_Conv(nn.Module):
    ### only works with square image, kernel size and stride
    def __init__(self, base_channel_size: int, net_type : Type[cl.interfaceModule] , image_size :int, out_size = 10 ) -> None:
        super().__init__()
        
        alpha = base_channel_size
        channel_size :int = 3
        im_size : int = image_size
        
        modules : List[nn.Module] = []
        size_conv = [
            (alpha,1),
            (2*alpha,2), #remove stride for now
            (2*alpha,1),
            (4*alpha,2), #here too
            (4*alpha,1),
            (8*alpha,2), # here too
            (8*alpha,1),
            (16*alpha,2)] # here too
        
        for i,val in enumerate(size_conv):
            size_out, stride = val
            im_size,channel_size,new_module = net_type(channel_size,size_out,3,stride,im_size).get_module()
            modules.append(new_module)
            
        self.conv = nn.Sequential(*modules)
        im_size,channel_size,self.fc = cl.FC_custom(channel_size,64*alpha,0,1,im_size).get_module()
        self.fc_final = nn.Linear(channel_size*im_size**2,out_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        x = self.fc_final(x)
        return x

class S_Conv(nn.Module):
    def __init__(self, base_channel_size: int, net_type :Type[cl.interfaceModule], image_size :int,out_size = 10) -> None:
        super().__init__()
        
        alpha = base_channel_size
        channel_size :int = 3
        im_size : int = image_size
        
        im_size,channel_size, self.conv = net_type(channel_size,alpha,9,2,im_size).get_module()
        im_size,channel_size, self.fc = cl.FC_custom(channel_size,24*alpha,0,1,im_size).get_module()
        self.fc_final = nn.Linear(channel_size*im_size**2,out_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        x = self.fc_final(x)
        return x