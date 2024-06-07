Underwater Image Enhancement using Deep Residual Framework
Project Description
This MATLAB project aims to enhance underwater images using a Very Deep Super Resolution (VDSR) neural network model. The project includes the creation of training data, network architecture, and evaluation of enhanced images using metrics such as PSNR, BRISQUE, and MAE.

Directory Structure
bash
Copy code
/MATLAB Drive/CODE/CODE/
│
├── upsampledImages/      # Directory for storing upsampled images
├── residualImages/       # Directory for storing residual images
└── ...
Requirements
MATLAB with Deep Learning Toolbox
Image Processing Toolbox
Usage
Clone the repository to your local machine.
Navigate to the project directory in MATLAB.
Run the main script to select and enhance an underwater image.
Training the Network
The VDSR model is trained using a dataset of upsampled images and corresponding residuals. The training process involves defining the network architecture, setting training options, and using the trainNetwork function.

Evaluation Metrics
The enhanced images are evaluated using the following metrics:

PSNR (Peak Signal-to-Noise Ratio): Measures the quality of the enhanced image compared to the original.
BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): Provides a no-reference image quality score.
MAE (Mean Absolute Error): Calculates the average absolute differences between the original and enhanced images.
Results
The project displays the input, low-resolution, and enhanced images. The PSNR, BRISQUE, and MAE values are also printed to the MATLAB console.

Contributing
Contributions to this project are welcome. Please submit a pull request or open an issue to discuss potential changes or improvements.
