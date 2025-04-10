from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import os
from torch.autograd import Function
from spikingjelly.clock_driven import neuron
from spikingjelly.clock_driven import surrogate
import math
import torch.utils.data as data
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cuda:3"

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class ScaledNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):
        #print(x.shape)
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.5)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale
    def reset(self):
        self.t = 0
        self.neuron.reset()

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class ShiftNeuron(nn.Module):
    def __init__(self, scale=1., alpha=1/50000):
        super().__init__()
        self.alpha = alpha
        self.vt = 0.
        self.scale = scale
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):
        x = x / self.scale
        x = self.neuron(x)
        return x * self.scale
    def reset(self):
        if self.training:
            self.vt = self.vt + self.neuron.v.reshape(-1).mean().item()*self.alpha
        self.neuron.reset()
        if self.training == False:
            self.neuron.v = self.vt

class MyFloor(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        print('up:',self.up)
        print('t:',self.t)

    def forward(self, x):
        #print(x.shape)
        x = x / self.up
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

class TCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
    def forward(self, x):
        x = F.relu(x, inplace='True')
        x = self.up - x
        x = F.relu(x, inplace='True')
        x = self.up - x
        return x

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


##混合训练方法在直接训练SNN上进行更改激活函数,需要改造适配于IF神经元
class LIFSpike(nn.Module):
    def __init__(self, scale=1.0,thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.scale=scale

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[0]
        x=x/self.scale
        for t in range(T):
            mem = mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
            #print(torch.stack(spike_pot, dim=1).shape)
        x=x*self.scale
        return torch.stack(spike_pot, dim=0)

##改造的IF神经元结构
class IFSpike(nn.Module):
    def __init__(self, scale=1.0,thresh=1.0,gama=1.0):
        super(IFSpike, self).__init__()
        self.act = ZIF.apply
        self.scale = scale
        self.gama = gama
        self.thresh=thresh

    def forward(self, x):
        spike_pot = []
        T = x.shape[0]
        mem = 0.5
        x=x/self.scale
        out=2
        for t in range(T):
            mem = mem + x[t ,...]
            #print('mem',mem)
            spike = self.act(mem -self.thresh, self.gama)
            #print(spike)
            #mem = (1 - spike) * mem
            mem=mem-spike*self.thresh
            spike_pot.append(spike)
        x=x*self.scale
        return torch.stack(spike_pot,dim=0)


# 改造的IF神经元2.0版本
class IFSpike1(nn.Module):
    def __init__(self, scale=1.0,thresh=1.0,alpha=2.0):
        super(IFSpike1, self).__init__()
        self.act = atan.apply
        self.scale = scale
        self.alpha = alpha
        self.thresh=thresh

    def forward(self, x):
        spike_pot = []
        T = x.shape[0]
        mem = 0.5
        x=x/self.scale
        for t in range(T):
            mem = mem + x[t ,...]
            #print('mem',mem)
            spike = self.act(mem -self.thresh, self.alpha)
            #print(spike)
            #mem = (1 - spike) * mem
            mem=mem-spike*self.thresh
            spike_pot.append(spike)
        x=x*self.scale
        return torch.stack(spike_pot,dim=0)


##更改转换后的SNN继续训练的激活函数，参考TET
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None