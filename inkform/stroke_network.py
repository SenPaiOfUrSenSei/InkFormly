"""Inference architecture for the independently trained Vasquez stroke weights.

Implements three standard LSTM cells, a moving Gaussian text window, and a
mixture-density head. The weights are imported from the documented PyTorch
conversion of Sean Vasquez's TensorFlow checkpoint; no TensorFlow dependency
or upstream application code is needed at runtime.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


ALPHABET = '\0 !"#\'(),-.0123456789:;?ABCDEFGHIJKLMNOPRSTUVWYabcdefghijklmnopqrstuvwxyz'


@dataclass
class PenState:
    hidden: tuple[torch.Tensor, ...]
    memory: tuple[torch.Tensor, ...]
    position: torch.Tensor
    window: torch.Tensor
    attention: torch.Tensor | None = None


class StrokeNetwork(nn.Module):
    def __init__(self, hidden=400, windows=10, mixtures=20):
        super().__init__()
        self.hidden, self.windows, self.mixtures = hidden, windows, mixtures
        vocabulary = len(ALPHABET)
        self.layers = nn.ModuleList([nn.LSTMCell(vocabulary+3, hidden),
                                     nn.LSTMCell(vocabulary+3+hidden, hidden),
                                     nn.LSTMCell(vocabulary+3+hidden, hidden)])
        self.alignment = nn.Linear(vocabulary+3+hidden, windows*3)
        self.density = nn.Linear(hidden, mixtures*6+1)
        self.t_adapter = None

    def import_weights(self, weights):
        mapped = {}
        for i in range(3):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                mapped[f"layers.{i}.{name}"] = weights[f"cell.lstm{i+1}.{name}"]
        for name in ("weight", "bias"):
            mapped[f"alignment.{name}"] = weights[f"cell.attn_linear.{name}"]
            mapped[f"density.{name}"] = weights[f"cell.gmm_linear.{name}"]
        self.load_state_dict(mapped, strict=True)

    def initial_state(self):
        return PenState(tuple(torch.zeros(1,self.hidden) for _ in range(3)),
                        tuple(torch.zeros(1,self.hidden) for _ in range(3)),
                        torch.zeros(1,self.windows), torch.zeros(1,len(ALPHABET)))

    def step(self, pen, state, text):
        h0,c0 = self.layers[0](torch.cat((state.window,pen),-1),(state.hidden[0],state.memory[0]))
        a,b,advance = F.softplus(self.alignment(torch.cat((state.window,pen,h0),-1))).chunk(3,-1)
        position = state.position + advance/25
        slots = torch.arange(text.shape[1],dtype=torch.float32).reshape(1,1,-1)
        attention = (a.unsqueeze(-1)*torch.exp(-(position.unsqueeze(-1)-slots).square()/b.clamp_min(.01).unsqueeze(-1))).sum(1)
        window = (attention.unsqueeze(-1)*text).sum(1)
        h1,c1 = self.layers[1](torch.cat((pen,h0,window),-1),(state.hidden[1],state.memory[1]))
        h2,c2 = self.layers[2](torch.cat((pen,h1,window),-1),(state.hidden[2],state.memory[2]))
        return PenState((h0,h1,h2),(c0,c1,c2),position,window,attention)

    def distribution(self, state, bias, form_diversity=0.0, t_focus=0.0):
        k = self.mixtures
        raw = self.density(state.hidden[2])
        if t_focus and self.t_adapter is not None:
            raw = raw + t_focus*self.t_adapter(state.hidden[2])
        logits,log_sigma,correlation,mean,pen_logit = raw.split((k,2*k,k,2*k,1),-1)
        # Broaden the choice of *learned* next movements independently of
        # coordinate noise. Components are local motions, not labeled letters
        # or interchangeable writer identities. Zero preserves old sampling.
        temperature = 1 + .65 * form_diversity
        probability = F.softmax(logits*(1+bias)/temperature,-1)
        probability = torch.where(probability<.01,0,probability)
        probability /= probability.sum(-1,keepdim=True)+1e-8
        sigma = torch.exp(log_sigma-bias).clamp_min(1e-4)
        rho = correlation.tanh().clamp(-1+1e-8,1-1e-8)
        lift = pen_logit.sigmoid().clamp(1e-8,1-1e-8)
        lift = torch.where(lift<.01,0,lift)
        return probability,mean,sigma,rho,lift

    def draw(self, state, bias, form_diversity=0.0, t_focus=0.0):
        probability,mean,sigma,rho,lift = self.distribution(state,bias,form_diversity,t_focus)
        component = int(torch.multinomial(probability[0],1))
        k = self.mixtures
        a,b = torch.randn(2)
        r = rho[0,component]
        x = mean[0,component] + sigma[0,component]*a
        y = mean[0,component+k] + sigma[0,component+k]*(r*a+torch.sqrt(1-r*r)*b)
        pen = torch.bernoulli(lift[0,0])
        return torch.stack((x,y,pen)).reshape(1,3)

    @staticmethod
    def t_focus_profile(text, content_start=0, include_uppercase=True):
        """Soft sampling focus for t/T and adjacent joins, not glyph geometry.

        The primer is excluded. Internal/final letters receive more focus than
        a word-initial t; neighboring characters cover entry and exit motion.
        Text attention blends these weights as the learned pen moves.
        """
        profile = torch.zeros(len(text)+1)
        for index in range(content_start,len(text)):
            if text[index] not in ('tT' if include_uppercase else 't'):continue
            weight = 1. if index>content_start and text[index-1].isalpha() else .45
            profile[index] = max(float(profile[index]),weight)
            for offset,multiplier in ((-1,.25),(1,.5),(2,.2)):
                neighbor = index+offset
                if content_start<=neighbor<len(text) and text[neighbor].isalpha():
                    profile[neighbor] = max(float(profile[neighbor]),weight*multiplier)
        return profile

    @torch.inference_mode()
    def sample(self, text, primer, seed, bias, max_steps, form_diversity=0.0,
               t_clarity=0.0, content_start=0, t_strategy='adapter'):
        if not 0 <= form_diversity <= 1:
            raise ValueError("Form diversity must be between 0 and 1.")
        if not 0 <= t_clarity <= 1:
            raise ValueError("T clarity must be between 0 and 1.")
        if not 0<=content_start<len(text):
            raise ValueError("Content start must be inside the supplied text.")
        if t_strategy not in ('adapter','precision'):
            raise ValueError('Unknown t sampling strategy.')
        if t_clarity and t_strategy=='adapter' and self.t_adapter is None:
            raise ValueError('The learned t adapter has not been loaded.')
        context = F.one_hot(torch.tensor([[ALPHABET.index(c) for c in text]+[0]]),len(ALPHABET)).float()
        state = self.initial_state()
        focus_profile = self.t_focus_profile(text,content_start,include_uppercase=t_strategy=='precision') if t_clarity else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if primer is not None:
                for point in primer:
                    state = self.step(point.reshape(1,3),state,context)
            pen = torch.tensor([[0.,0.,1.]])
            result = []
            completed = False
            for _ in range(max_steps):
                state = self.step(pen,state,context)
                # Use the output draw for both continuation and termination.
                # A pen lift alone cannot terminate a character or sentence.
                local_bias,local_diversity,focus = bias,form_diversity,0.
                if focus_profile is not None:
                    weights = state.attention.reshape(-1)
                    focus = t_clarity*float((weights*focus_profile).sum()/weights.sum().clamp_min(1e-8))
                    if focus>0 and t_strategy=='precision':
                        local_bias = min(1.2,bias+.3*focus)
                        local_diversity = form_diversity*(1-.65*focus)
                pen = self.draw(state,local_bias,local_diversity,focus) if focus and t_strategy=='adapter' else self.draw(state,local_bias,local_diversity)
                if not torch.isfinite(pen).all() or pen[0,:2].abs().max()>100:
                    break
                # Terminal punctuation may otherwise grow extra strokes after
                # it is finished. Letters/numbers need the stricter end marker
                # so a late crossbar or dot can still be drawn. The learned
                # attention window and all generated movements stay unchanged.
                attention = state.attention.reshape(-1)
                end_strength = 1.0 if text[-1].isalnum() else .5
                final = (int(attention.argmax()) >= len(text)-1 and
                         float(attention[-1]) >= end_strength*float(attention.max()))
                # A lift ends the segment *at this coordinate*. Retain it so
                # the final letter/mark is not shortened by one pen movement.
                result.append(pen[0].clone())
                if final and pen[0,2]>.5:
                    completed = True
                    break
        if not result:
            return None,False
        values = torch.stack(result)
        values[-1,2] = 1
        return values,completed
