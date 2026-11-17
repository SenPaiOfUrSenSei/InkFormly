"""Measure repeated forms in UCI Character Trajectories, one recorded writer.

Requires the optional research dependency scipy. Dataset is never substituted
into the generator: this writer is different from InkForm's chosen reference.
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]


def resample(points, count=96):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    keep = np.r_[True, np.diff(distance)>1e-10]
    distance, points = distance[keep], points[keep]
    t = np.linspace(0.,distance[-1],count)
    values = np.column_stack([np.interp(t,distance,points[:,axis]) for axis in (0,1)])
    values -= values.mean(0)
    return values / max(np.sqrt(np.mean(np.sum(values**2,axis=1))),1e-10)


def similarity_residual(a,b):
    """Remove rotation and a uniform scale, retaining proportions and form."""
    u,_,vt = np.linalg.svd(a.T@b)
    correction = np.diag([1.,np.linalg.det(u@vt)])
    rotated = a@(u@correction@vt)
    scale = np.sum(rotated*b)/np.sum(rotated**2)
    return float(np.sqrt(np.mean(np.sum((rotated*scale-b)**2,axis=1))))


def main():
    archive = ROOT/'data/character-trajectories/character-trajectories.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != '5d2db017ef0d8cf0e65ed060c9e90399f78eb9f1e3cb63e22ca8c3ef4ba67d52':
        raise ValueError('Dataset archive checksum differs from the analyzed release.')
    with ZipFile(archive) as z:
        data = loadmat(BytesIO(z.read('mixoutALL_shifted.mat')),simplify_cells=True)
    out = ROOT/'artifacts/evaluation/human-variation'
    out.mkdir(parents=True,exist_ok=True)
    consts = data['consts']
    norm = np.asarray(consts['datanorm'])[:2]
    labels = np.asarray(consts['charlabels'])
    rng = np.random.default_rng(73)
    rows, report = {}, {}
    for label,character in enumerate(consts['key'],1):
        ids = np.flatnonzero(labels==label)
        selected = rng.choice(ids,min(60,len(ids)),replace=False)
        forms = [resample(np.cumsum(np.asarray(data['mixout'][i])[:2].T*norm,axis=0)) for i in selected]
        # Pair sample without sorting for most/least similar examples.
        residuals = [similarity_residual(a,b) for a,b in zip(forms[::2],forms[1::2])]
        report[str(character)] = dict(total=int(len(ids)),sampled_pairs=len(residuals),
            similarity_aligned_rms_p10_p50_p90=np.quantile(residuals,[.1,.5,.9]).tolist())
        rows[str(character)] = forms[:8]
    sheet = Image.new('RGB',(1400,1000),'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((20,15),'Recorded letters: eight samples each, same human writer (UCI Character Trajectories)',fill='black')
    draw.text((20,35),'Arc-length resampled, centered, uniform size normalized; no shape deformation or glyph replacement.',fill='black')
    for row,character in enumerate('aegmns'):
        draw.text((20,100+row*145),character,fill='black')
        for col,form in enumerate(rows[character]):
            p = form.copy()
            p[:,1] *= -1
            p = p*30+[120+col*160,130+row*145]
            draw.line([tuple(point) for point in p],fill='#203f78',width=2,joint='curve')
    draw.text((20,965),'Ben Williams (2006), doi:10.24432/C58G7V, CC BY 4.0. Processed from recorded pen velocities.',fill='black')
    sheet.save(out/'recorded-letter-variation.png')
    (out/'measurements.json').write_text(json.dumps(dict(
        source='https://archive.ics.uci.edu/dataset/175/character%2Btrajectories',
        samples=int(len(labels)),writers=1,seed=73,letters=report,
        method='Integrate xy velocities after undoing per-axis normalization; resample 96 equal arc-length points, center/RMS normalize, pairwise similarity align. Residual includes proportion and shape variation.',
        limitations='Isolated single-pen-down characters, smoothed and temporally shifted. Different writer from model primer. Not a cursive/coarticulation dataset or a calibration of neural temperature; no pressure inference or training.'),indent=2))
    print(out)


if __name__=='__main__':
    main()
