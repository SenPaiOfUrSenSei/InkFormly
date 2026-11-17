# Vendored Graves synthesis architecture

Source: https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit

Revision: `899e9432dfe0998ba2ef31582d11c2d5f24a5185`

`models.py` is the upstream neural network architecture, distributed under the
adjacent MIT license. The only local modification is replacing the import-time
file logger with a `NullHandler`. InkForm handles input validation, inference,
sampling and rendering outside this file. No training toolkit dependencies are
needed to run the model.

The pretrained checkpoints originate in the upstream `checkpoints` directory.
The repository license does not settle separate IAM dataset rights; do not assume
that the checkpoint is cleared for every commercial use.
