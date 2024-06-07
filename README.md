**Underwater Image Enhancement using Deep Residual Framework**

**Project Description**

Underwater images often suffer from degradation due to light refraction and absorption, affecting visibility and color balance. This MATLAB project aims to enhance such images using a Very Deep Super Resolution (VDSR) neural network model within a deep residual learning framework. The project involves creating training data, defining network architecture, and evaluating enhanced images using metrics such as PSNR, BRISQUE, and MAE

**Requirements**

- MATLAB online or MATLAB R2013a

**Usage**

1. Clone the repository to your local machine.
2. Navigate to the project directory in MATLAB.
3. Run the main script to select and enhance an underwater image.

**Key Concepts**

- **Cycle GAN**: Generates synthetic underwater images for additional training data.
- **VDSR (Very-Deep Super-Resolution Reconstruction Model)**: Enhances image quality for underwater applications.
- **Underwater Resnet**: A tailored residual learning architecture for improving underwater images.

**Enhancements and Modifications**

- **Multi-Term Loss Function**: Combines mean squared error loss with edge difference loss for superior color correction and detail enhancement.
- **Asynchronous Training Mode**: Optimizes the multi-term loss function asynchronously.
- **Batch Normalization**: Explored for its impact on model performance.

**Training the Network**

The VDSR model is trained using a dataset of upsampled images and corresponding residuals. The training process involves defining the network architecture, setting training options, and using the trainNetwork function.

**Evaluation Metrics**

The enhanced images are evaluated using the following metrics:

- PSNR (Peak Signal-to-Noise Ratio): Measures the quality of the enhanced image compared to the original.
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): Provides a no-reference image quality score.
- MAE (Mean Absolute Error): Calculates the average absolute differences between the original and enhanced images.

**Results**

The project displays the input, low-resolution, and enhanced images. The PSNR, BRISQUE, and MAE values are also printed to the MATLAB console.

**Contributing**

Contributions to this project are welcome. Please submit a pull request or open an issue to discuss potential changes or improvements.
