# DOT_filtering
An automated pipeline for breast cancer diagnosis using US-assisted diffuse optical tomography.

**by Minghao Xue (https://opticalultrasoundimaging.wustl.edu/)**

## Abstract

Traditional diffuse optical tomography (DOT) reconstructions suffer from image artifacts due to various factors such as the proximity of DOT sources to shallow lesions, poor optode-tissue coupling, tissue heterogeneity, and large high-contrast lesions causing shadowing effects. This study introduces an attention-based U-Net (APU-Net) model with Contextual Transformer (CoT) attention modules to enhance DOT image quality and improve lesion diagnostic accuracy. Trained on simulation and phantom data, and evaluated on clinical data, the APU-Net model effectively reduced artifacts by 26.83% on average and significantly improved image contrast in deeper regions, with increases of 20.28% and 45.31% for the second and third target layers, respectively. These improvements demonstrate the potential of the APU-Net model in enhancing DOT reconstructions for better breast cancer diagnosis.

![Structure](https://github.com/OpticalUltrasoundImaging/DOT_filtering/blob/main/images/structure.tif)

## Requirements
* Python: 3.7+
* torch(pytorch): 1.10+
* torchvision: 0.11.1+
* numpy: 1.21.2+
* scipy: 1.7.1+
* scikit-learn: 1.3+


## Contact

Please email Minghao Xue at m.xue@wustl.edu if you have any concerns or questions.
