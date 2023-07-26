import torch
import torch.nn as nn

"""
使い方
ModelBaseを継承してinit_class()で定義することによりdiffusion.pyで使用できます．
"""

def init_class():
    model_map={
        "FFNModel":FFNModel,
        "FFN3Model":FFN3Model,
        "DeepFFNModel":DeepFFNModel,
    }
    return model_map

class ModelBase(nn.Module):
    pass

class FFNModel(ModelBase):
    def __init__(self,dim) -> None:
        super(FFNModel, self).__init__()
        self.ffn1=nn.Linear(dim,5)
        self.sig=nn.Sigmoid()
        self.ffn2=nn.Linear(5,dim)

    def forward(self, x, t) -> torch.Tensor:
        x=self.ffn1(x)
        x=self.sig(x)
        x=self.ffn2(x)
        return x

class FFN3Model(ModelBase):
    def __init__(self,dim) -> None:
        super(FFN3Model, self).__init__()
        self.ffn1=nn.Linear(dim,5)
        self.sig=nn.Sigmoid()
        self.ffn2=nn.Linear(5,dim)

    def forward(self, x, t) -> torch.Tensor:
        x=self.ffn1(x)
        x=self.sig(x)
        x=self.ffn2(x)
        x=x**3
        return x

class DeepFFNModel(ModelBase):
    def __init__(self,dim) -> None:
        super(DeepFFNModel, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim,5),
            nn.Sigmoid(),
            nn.Linear(5,10),
            nn.Sigmoid(),
            nn.Linear(10,20),
            nn.Sigmoid(),
            nn.Linear(20,10),
            nn.Sigmoid(),
            nn.Linear(10,5),
            nn.Sigmoid(),
            nn.Linear(5,dim),
        )

    def forward(self, x, t) -> torch.Tensor:
        return self.ffn(x)

model_map=init_class()