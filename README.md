# DeepPyramidPlus_IJCARS
This repository provides the official PyTorch implementation of DeepPyramid+ (Pyramid View Fusion and Deformable Pyramid Reception), the invited paper at IJCARS' special session for MICCAI 2022.

DeepPyramid+: Medical Image Segmentation using Pyramid View Fusion and Deformable Pyramid Reception
Authors:
Negin Ghamsarian<sup>1</sup>, Sebastian Wolf<sup>3</sup>, Martin Zinkernagel<sup>3</sup>, Klaus Schoeffmann<sup>2</sup>, Raphael Sznitman<sup>1</sup>
<sup>1</sup>ARTORG Center for Biomedical Engineering Research, University of Bern, Bern, Switzerland
<sup>2</sup>Department of Information Technology, University of Klagenfurt, Klagenfurt, Austria
<sup>3</sup>Department of Ophthalmology, Inselspital, Bern, Switzerland

Published in: International Journal of Computer Assisted Radiology and Surgery (2024)
DOI: https://doi.org/10.1007/s11548-023-03046-2

Overview
DeepPyramid+ is a novel neural network architecture designed for medical image and surgical video segmentation. This paper addresses the challenges encountered in segmenting heterogeneous, deformable, and transparent objects in medical images and videos. By incorporating two key modules, Pyramid View Fusion (PVF) and Deformable Pyramid Reception (DPR), the model improves segmentation accuracy and robustness.

Key Features
Pyramid View Fusion (PVF): Enhances representation by mimicking a human-like deduction process within the network, which improves pixel-level information extraction.
Deformable Pyramid Reception (DPR): Introduces dilated deformable convolutions for adaptive shape- and scale-sensitive feature extraction.
Results
DeepPyramid+ was evaluated on several datasets, including endometriosis videos, MRI, OCT, cataract surgery, and laparoscopy videos. The model demonstrated superior performance with improvements of up to 3.65% in the Dice coefficient for intra-domain segmentation and 17% for cross-domain tasks compared to state-of-the-art models.

Datasets
Experiments were conducted on various datasets from different modalities:

Cataract Surgery Instruments
Laparoscopy Instruments
Endometriosis Implants
Prostate MRI
Retina OCT
Cross-Domain Cataract and Prostate MRI
Installation and Usage
Prerequisites
Python 3.8+
PyTorch 1.9.0+
Clone the repository
bash
Copy code
git clone https://github.com/Negin-Ghamsarian/DeepPyramid_Plus.git
cd DeepPyramid_Plus
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Usage
Download and prepare the datasets.
Run the following command to train the model:
bash
Copy code
python train.py --config config.yaml
Use the provided scripts for evaluating the performance on the test set.
License
This work is licensed under a Creative Commons Attribution 4.0 International License.

Citation
If you find this work helpful, please cite the paper:

sql
Copy code
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
This README provides a succinct summary of your work, including sections on features, results, and instructions for installation and usage. Let me know if you'd like to add or modify anything!
```

## Acknowledgments

This work was funded by Haag-Streit Switzerland.
