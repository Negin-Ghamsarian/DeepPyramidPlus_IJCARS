# DeepPyramidPlus_IJCARS
This repository provides the official PyTorch implementation of DeepPyramid+ (Pyramid View Fusion and Deformable Pyramid Reception), the invited paper at IJCARS' special session for MICCAI 2022.

## Overview

DeepPyramid+ is a novel neural network architecture designed for medical image and surgical video segmentation. This model addresses the key challenges of segmenting heterogeneous, deformable, and transparent objects in medical images and videos. The architecture incorporates two main modules: Pyramid View Fusion (PVF) and Deformable Pyramid Reception (DPR), which significantly enhance segmentation accuracy and robustness.

## Key Features

- **Pyramid View Fusion (PVF):** A module that mimics a human-like deduction process within the network, improving pixel-wise information extraction.
- **Deformable Pyramid Reception (DPR):** Introduces dilated deformable convolutions for adaptive shape- and scale-sensitive feature extraction, improving robustness against shape and size variations.

## Results

The model was tested on several datasets, including endometriosis videos, MRI, OCT, cataract surgery, and laparoscopy videos. Results showed a Dice coefficient improvement of up to 3.65% for intra-domain segmentation and up to 17% for cross-domain segmentation tasks, outperforming other state-of-the-art methods.

## Datasets

DeepPyramid+ has been evaluated on multiple medical imaging datasets, including:

1. **Cataract Surgery Instruments**  
2. **Laparoscopy Instruments**  
3. **Endometriosis Implants**  
4. **Prostate MRI**  
5. **Retina OCT**  
6. **Cross-Domain Cataract and Prostate MRI**

## Installation and Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.9.0+

### Clone the repository

```bash
git clone https://github.com/Negin-Ghamsarian/DeepPyramid_Plus.git
cd DeepPyramid_Plus
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Training
Download the datasets and prepare them for training.
To train the model, use the following command:
bash
Copy code
python train.py --config config.yaml
Evaluation
Use the provided evaluation scripts to measure performance on the test set:

bash
Copy code
python evaluate.py --checkpoint best_model.pth --dataset test_dataset
License
This work is licensed under a Creative Commons Attribution 4.0 International License.

Citation
If you use this work in your research, please cite it as follows:

@article{ghamsarian2024deeppyramid,
  title={DeepPyramid+: medical image segmentation using Pyramid View Fusion and Deformable Pyramid Reception},
  author={Ghamsarian, Negin and Wolf, Sebastian and Zinkernagel, Martin and Schoeffmann, Klaus and Sznitman, Raphael},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={19},
  number={6},
  pages={851--859},
  year={2024},
  publisher={Springer}
}

## Acknowledgments

This work was funded by Haag-Streit Switzerland.
