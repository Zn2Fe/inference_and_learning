import torch
import torch.nn as nn
import networks.custom_layers as cl

from typing import Type,List

class D_Conv(nn.Module):
    ### only works with square image, kernel size and stride
    def __init__(        self,
        base_channel_size: int,
        net_type :Type[cl.interfaceModule],
        image_size :int,
        out_size = 10,
        dropout_1 = True,
        dropout_2 = True ) -> None:
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
        self.fc = nn.Linear(channel_size*im_size**2,24*alpha)
        if dropout_1:
            self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(24*alpha,out_size)
        if dropout_2:
            self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.fc_final(x)
        if hasattr(self, 'dropout2'):
            x = self.dropout2(x)

class S_Conv(nn.Module):
    def __init__(
        self,
        base_channel_size: int,
        net_type :Type[cl.interfaceModule],
        image_size :int,
        out_size = 10,
        dropout_1 = True,
        dropout_2 = True) -> None:
        super().__init__()
        
        alpha = base_channel_size
        channel_size :int = 3
        im_size : int = image_size
        
        im_size,channel_size, self.conv = net_type(channel_size,alpha,9,2,im_size).get_module()
        self.fc = nn.Linear(channel_size*im_size**2,24*alpha)
        if dropout_1:
            self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(24*alpha,out_size)
        if dropout_2:
            self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.fc_final(x)
        if hasattr(self, 'dropout2'):
            x = self.dropout2(x)
        return x