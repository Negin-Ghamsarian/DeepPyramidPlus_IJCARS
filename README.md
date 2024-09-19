# DeepPyramidPlus_IJCARS
This repository provides the official PyTorch implementation of DeepPyramid+ (Pyramid View Fusion and Deformable Pyramid Reception), the invited paper at IJCARS' special session for MICCAI 2022.

## Overview

DeepPyramid+ is a novel neural network architecture designed for medical image and surgical video segmentation. This model addresses the key challenges of segmenting heterogeneous, deformable, and transparent objects in medical images and videos. The architecture incorporates two main modules: Pyramid View Fusion (PVF) and Deformable Pyramid Reception (DPR), which significantly enhance segmentation accuracy and robustness.

## Key Features

- **Pyramid View Fusion (PVF):** A module that mimics a human-like deduction process within the network, improving pixel-wise information extraction.
- **Deformable Pyramid Reception (DPR):** Introduces dilated deformable convolutions for adaptive shape- and scale-sensitive feature extraction, improving robustness against shape and size variations.

<img src="./Figures/BD_rev.pdf" alt="Proposed Pyramid View Fusion and Deformable Pyramid Reception modules." width="500">
<img src="./Figures/modules_rev.pdf" alt="Proposed Pyramid View Fusion and Deformable Pyramid Reception modules." width="500">

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

<img src="./Figures/Datasets.pdf" alt="Proposed Pyramid View Fusion and Deformable Pyramid Reception modules." width="500">

## Citation
If you use this work in your research, please cite it as follows:

```
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
```

## Acknowledgments

This work was funded by Haag-Streit Switzerland.
