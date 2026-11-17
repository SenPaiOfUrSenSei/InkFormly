"""Small residual motion-density adapter; the pretrained recurrent model stays frozen."""
import torch
from torch import nn


class TStrokeAdapter(nn.Module):
    def __init__(self,hidden=400,rank=16,outputs=121):
        super().__init__()
        self.down=nn.Linear(hidden,rank)
        self.up=nn.Linear(rank,outputs)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self,hidden):
        # A bounded residual prevents large changes to mixture or pen logits.
        return .5*torch.tanh(self.up(torch.tanh(self.down(hidden))))


def motion_nll(raw,targets,mixtures=20):
    """Unbiased bivariate Gaussian-mixture likelihood and pen-lift BCE."""
    k=mixtures
    logits,logsigma,rho,mean,lift=raw.split((k,2*k,k,2*k,1),-1)
    sx,sy=logsigma.clamp(-7,5).exp().chunk(2,-1)
    mx,my=mean.chunk(2,-1)
    r=rho.tanh().clamp(-.995,.995)
    x=(targets[:,0:1]-mx)/sx;y=(targets[:,1:2]-my)/sy
    covariance=(1-r.square()).clamp_min(1e-5)
    logpdf=-torch.log(2*torch.pi*sx*sy)-.5*torch.log(covariance)-(x.square()+y.square()-2*r*x*y)/(2*covariance)
    coordinate=-torch.logsumexp(torch.log_softmax(logits,-1)+logpdf,-1)
    pen=nn.functional.binary_cross_entropy_with_logits(lift[:,0],targets[:,2],reduction='none')
    return coordinate+pen
