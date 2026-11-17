# Pretrained stroke network

Original trained network: Sean Vasquez,
https://github.com/sjvasquez/handwriting-synthesis

PyTorch weight conversion used for local inference:
https://github.com/niccolodevries/svg-handwriting-generator
revision `61544f003893c99cbcb9013b33d0a437bbe1ac6f`,
`synth/checkpoints/model.pt`.

Checkpoint identity: original training step 17900; 3,637,231 parameters.
The fixed default writer is reference 2 (`House was crowded.`) from the upstream
IAM style examples. The earlier upright reference 0 (`thought that vengeance`) and rounded reference 3 (`A stronger finish`) is
also retained for comparisons. This is an actual reference trajectory, not a manually
constructed alphabet. Downloads and SHA-256 checks are recorded in
`scripts/download_readable_model.py`.

InkForm implements the inference architecture in `inkform/stroke_network.py`:
three standard LSTM cells, moving Gaussian text attention, and a 20-component
bivariate mixture plus Bernoulli pen lift. No TensorFlow or upstream GUI/source
module is imported at runtime. Each generated point remains a learned sample;
no letter lookup, corrected-word cache, font, or manually drawn punctuation is
used by this backend.

The current t-word policy generates complete alternatives with the same network
and writer reference, and fits a selected whole word at verified pen lifts using
translation and uniform scale. It does not deform individual glyphs or crossbars.
The earlier geometric stem and crossbar edits were rejected after user review
and are excluded from generation. Base weights remain unchanged.
A separately trained small adapter (`t_candidate_adapter.pt`) is a
local experiment using existing upstream IAM style references; it is not loaded
by default because its held-out readbacks regressed. The production decoder
does not use those adapter weights or the experimental contrastive reranker.

The original model and conversion repositories do not provide an explicit
license grant in the inspected files. These downloaded assets are retained for
local evaluation, excluded from Git, and are not represented as MIT-licensed
InkForm assets. Redistribution rights remain to be established separately.
