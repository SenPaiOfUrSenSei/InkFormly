"""Train a bounded t-motion residual on existing recorded IAM style references.

This tiny local experiment is not broad handwriting pretraining. No generated
probe, essay or test sentence is used for fitting; held-out references are
separated by reference ID. All pretrained model parameters remain frozen.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from inkform.readable_generator import ReadableGenerator
from inkform.stroke_network import ALPHABET
from inkform.t_adapter import TStrokeAdapter,motion_nll

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/evaluation/t-adapter'
TRAIN=[0,1,3,5,7,9]
VALID=[10,11,12]


def features(model,ids):
    rows=[];report=[]
    with torch.inference_mode():
        for style in ids:
            folder=ROOT/'models/vasquez/styles'
            path=folder/f'style-{style}-strokes.npy'
            points=torch.tensor(np.load(path,allow_pickle=False),dtype=torch.float32)
            text=np.load(folder/f'style-{style}-chars.npy',allow_pickle=False).tobytes().decode()
            context=F.one_hot(torch.tensor([[ALPHABET.index(c) for c in text]+[0]]),len(ALPHABET)).float()
            profile=model.t_focus_profile(text,include_uppercase=False)
            state=model.initial_state();count=0
            for i in range(len(points)-1):
                state=model.step(points[i:i+1],state,context)
                attention=state.attention.reshape(-1)
                focus=(attention*profile).sum()/attention.sum().clamp_min(1e-8)
                if float(focus)<.08:continue
                rows.append((state.hidden[2][0].clone(),model.density(state.hidden[2])[0].clone(),points[i+1].clone(),focus.clone()))
                count+=1
            report.append(dict(reference=style,text=text,selected_steps=count,t_count=text.count('t'),capital_T_count=text.count('T'),
                final_attention_index=int(state.attention.argmax()),text_length=len(text),
                strokes_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Copy out of inference mode so the adapter can backpropagate through its
    # own parameters using these fixed input states.
    data=tuple(torch.stack([row[i] for row in rows]).clone() for i in range(4))
    return data,report


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(91027)
    g=ReadableGenerator(adapter_path=None)
    for parameter in g.model.parameters():parameter.requires_grad_(False)
    train,train_report=features(g.model,TRAIN);valid,valid_report=features(g.model,VALID)
    print(json.dumps(dict(train_steps=len(train[0]),validation_steps=len(valid[0]),train=train_report,validation=valid_report)),flush=True)
    adapter=TStrokeAdapter()
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=8e-4,weight_decay=.01)
    def score(data):
        h,raw,target,focus=data
        delta=adapter(h)
        nll=motion_nll(raw+focus[:,None]*delta,target)
        return (nll*focus).sum()/focus.sum(),delta
    with torch.no_grad():
        baseline=float(score(valid)[0])
    best=baseline;best_state={k:v.detach().clone() for k,v in adapter.state_dict().items()};history=[];best_epoch=0
    for epoch in range(1,241):
        adapter.train();optimizer.zero_grad()
        train_loss,delta=score(train)
        objective=train_loss+.1*delta.square().mean()
        objective.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
        optimizer.step()
        if epoch%5==0:
            adapter.eval()
            with torch.no_grad():validation=float(score(valid)[0])
            row=dict(epoch=epoch,train_nll=float(train_loss.detach()),validation_nll=validation)
            history.append(row)
            if validation<best:
                best=validation;best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in adapter.state_dict().items()}
            if epoch%40==0:print(json.dumps(row),flush=True)
    checkpoint=ROOT/'models/vasquez/t_adapter-experimental.pt'
    torch.save(best_state,checkpoint)
    report=dict(seed=91027,target='lowercase t',parameters=sum(p.numel() for p in adapter.parameters()),
        train_references=train_report,validation_references=valid_report,
        baseline_validation_nll=baseline,best_validation_nll=best,best_epoch=best_epoch,
        history=history,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        limitations='Very small multi-writer reference set, mostly lowercase t. Reference-level held-out likelihood is not proof of legibility or writer consistency. Must pass independent synthesis comparisons before use.')
    (OUT/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','train_references','validation_references']}),flush=True)


if __name__=='__main__':main()
