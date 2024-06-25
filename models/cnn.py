
import torch.nn as nn
from pathlib import Path
import yaml
import torch


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    
    
    
class DetectModel(nn.Module):
    def __init__(self,cfg,nc=5,ch=()):
        super().__init__()
        self.nc=nc
                    
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding="ascii", errors="ignore") as f:
            self.yaml = yaml.safe_load(f)  # model dict
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        self.model=self.parseModel(self.yaml,[ch])
        m=self.model[-1]
        ch=m.conv.out_channels
        c=Classify(ch,nc)
        c.i,c.f,c.type=m.i,m.f,"Classify"
        self.model.append(c)
        self.save=[]
    def forward(self,x):
        y=[]
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x
        
    def parseModel(self,d,ch):
        gd=d["depth_multiple"]
        layers,c2=[],ch[-1]
        save=[]
        for i,(f,n,m,args) in enumerate(d["backbone"]):
            m=eval(m)
            n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

            if m in {Conv}:
                c1,c2=ch[f],args[0]
                args = [c1, c2, *args[1:]]
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            m_ = nn.Sequential(*(m(*args) for _ in range(n_))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i,m_.f,m_.type,m_.np=i,f,t,np
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
            
        
        return nn.Sequential(*layers)
    
class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
            
        
Model=DetectModel