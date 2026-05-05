# TRUST: Triple Space Alignment and Sharpness Tuning for Adversarial Robustness
---

## Requirements
First, install the required packages:
```bash
pip install torch torchvision
pip install tensorboard
pip install torchattacks
pip install autoattack
pip install numpy
pip install scipy
pip install tqdm
pip install pillow
pip install matplotlib
pip install pyhessian
pip install protobuf==3.20.3
```

- Python 3.8+
- PyTorch 1.10+
- CUDA support is recommended

---

## File Description
- `train.py`: Main training script for TRUST
- `evaluate.py`: Robustness evaluation script
- `trust_loss.py`: Core loss functions (tri_loss, trust_loss)
- `st_opt.py`: Sharpness Tuning optimizer (st_opt)

---

## Training
Run the following command to train the model:
```bash
python train.py
```

## Evaluation
Run the following command to evaluate the trained model:
```bash
python evaluate.py --checkpoint YOUR_CHECKPOINT_PATH
```
---
## Supported Attacks for Evaluation
- Clean (natural) accuracy
- FGSM
- PGD-20
- PGD-100
- CW
- AutoAttack

---
## Output
- Checkpoints: `results/`
- Evaluation logs: `robust_evaluate/`
