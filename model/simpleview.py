import torch
import torch.nn as nn

from model.resnet import resnet18_4
from model.resnet50_101 import ResNet50
import torch.nn.functional as F

class SimpleView(nn.Module):

    def __init__(self,):
        super().__init__()
        # self.backbone = resnet18_4()
        # z_dim = backbone.fc.in_features
        # backbone.fc = nn.Identity()
        backbone1 = ResNet50()
        # backbone2 = ResNet50()
        self.backbone1 = backbone1
        # self.backbone2 = backbone2
        #self.fc = nn.Linear(
        #    in_features=z_dim*num_views,
        #    out_features=num_classes
        #)
        # self.bn = nn.BatchNorm1d(1024)
        # self.dp = nn.Dropout(p=0.5)
        # self.backbone = ResNet50(num_classes=num_classes,channels=num_views)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x = x.reshape(b, c, h, w)
        # x1, x2 = torch.split(x, 10, dim=1)
        z = self.backbone1(x)
        # print("单视图特征维度:",z.shape) 
        # z = z.reshape(b,c, -1)    
        # print("单视图特征维度:",z.shape)   
        # z = z1 
        return  z
