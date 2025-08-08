# Generative-Inversion-for-Property-Targeted-Materials-Design-Application-to-Shape-Memory-Alloys
This repository contains the PyTorch implementation for a GAN inversion framework enabling property-targeted materials design, with specific application to Shape Memory Alloys (SMAs). The code implements conditional generation of materials composition-processing parameters with targeted properties via deep generative modeling.

## Repository Structure
```bash
.
├── c_p_WGAN_inversion.py            # Property-targeted inverse design framework
├── c_p_WGAN_gp_sampel.py            # Unconditional random sample generation
├── README.md
├── networks/
│   ├── ann_predict.py               # Predictor network architecture
│   ├── c_p_WGAN_gp.py               # WGAN-GP network architecture & training
│   ├── generator4096_94501_10_2_im_14_2.pth  # Pre-trained generator weights
│   ├── ann_2_512_2_3_1.pth          # Pre-trained predictor weights
│   └── ann_predict_final.py         # Predictor training strategy
├── tools/
│   ├── function.py                  # Data processing utilities
│   ├── score.py                     # Generator evaluation metrics
│   ├── dataloader.py                # Data loading functions
│   └── __init__.py
└── data/
    └── SME3.csv                     # SMA training dataset
```


## Key Components

### Core Modules
- **Generative Inversion** (`c_p_WGAN_inversion.py`):  
  Implements property-targeted material design via latent space optimization
- **Unconditional Sampling** (`c_p_WGAN_gp_sampel.py`):  
  Enables random microstructure generation from latent space distributions

### Networks
- `c_p_WGAN_gp.py`: Wasserstein GAN with Gradient Penalty architecture
- `ann_predict.py`: Multi-layer perceptron property predictor
- Pre-trained Models:
  - Generator: `generator4096_94501_10_2_im_14_2.pth`
  - Predictor: `ann_2_512_2_3_1.pth`

## Computational Environment
All implementations leverage GPU acceleration with:
- **Hardware**: NVIDIA T4 GPU
- **Software**:  
  - PyTorch 2.0.0
  - CUDA 11.7

## Reference
This work implements the methodology from:  
*Generative Inversion for Property-Targeted Materials Design: Application to Shape Memory Alloys* [Paper DOI/Citation Here]


