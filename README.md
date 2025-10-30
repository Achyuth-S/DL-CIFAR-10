# CNN Image Classification - CIFAR-10 Dataset

## Project Overview
This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is built using TensorFlow/Keras and achieves reasonable accuracy in classifying images into 10 different categories.

## Problem Statement
Image classification is a fundamental task in computer vision where the goal is to automatically categorize images into predefined classes. This project tackles the challenge of classifying 32x32 pixel color images from the CIFAR-10 dataset into one of 10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Dataset
**CIFAR-10** (Canadian Institute For Advanced Research)
- **Total Images:** 60,000 color images (32x32 pixels)
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images
- **Classes:** 10 categories with 6,000 images per class
- **Image Dimensions:** 32x32x3 (height × width × RGB channels)

### Class Labels
0. Airplane
1. Automobile
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Model Architecture

### CNN Structure
The implemented CNN architecture consists of:

1. **Input Layer:** 32x32x3 tensor (RGB images)

2. **Convolutional Block 1:**
   - Conv2D: 32 filters, 3x3 kernel, ReLU activation, same padding
   - MaxPooling2D: 2x2 pool size

3. **Convolutional Block 2:**
   - Conv2D: 64 filters, 3x3 kernel, ReLU activation, same padding
   - MaxPooling2D: 2x2 pool size

4. **Convolutional Block 3:**
   - Conv2D: 64 filters, 3x3 kernel, ReLU activation, same padding

5. **Fully Connected Layers:**
   - Flatten layer
   - Dense: 64 units, ReLU activation
   - Dropout: 0.5 (regularization to prevent overfitting)
   - Output Dense: 10 units, Softmax activation

### Key Features
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy
- **Training:** 15 epochs with batch size of 64
- **Validation Split:** 20% of training data

## Project Structure
```
CNN-CIFAR10-Image-Classification/
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
├── main.py                      # Main training script
├── saved_data/                  # Saved numpy arrays
│   ├── x_train.npy
│   ├── y_train.npy
│   ├── x_test.npy
│   └── y_test.npy
├── saved_model/                 # Saved model and results
│   ├── cifar10_cnn_model.h5
│   └── training_validation_accuracy.png
└── results/                     # Additional output files (optional)
```

## Installation & Setup

### Prerequisites
- Python 3.8 - 3.11
- pip or conda package manager

### Option 1: Using pip (General setup)
```bash
# Clone the repository
git clone https://github.com/yourusername/CNN-CIFAR10-Image-Classification.git
cd CNN-CIFAR10-Image-Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda (Recommended for macOS Apple Silicon M1/M2/M3)
```bash
# Create a new conda environment
conda create -n cifar10-cnn python=3.10 -y
conda activate cifar10-cnn

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon Macs, use:
conda install -c apple tensorflow-deps -y
pip install tensorflow-macos tensorflow-metal
pip install matplotlib numpy
```

### Troubleshooting macOS Installation Issues
If you encounter `Library not loaded: @rpath/libcblas.3.dylib` error:
```bash
conda clean --all -y
conda install -y openblas
conda uninstall -y numpy && conda install -y numpy
```

## Usage

### Running the Training Script
```bash
python main.py
```

This will:
1. Download and load the CIFAR-10 dataset
2. Preprocess and normalize the images
3. Build and compile the CNN model
4. Train the model for 15 epochs
5. Save the trained model to `saved_model/cifar10_cnn_model.h5`
6. Save training/validation accuracy plot
7. Save dataset arrays to `saved_data/` directory
8. Print test accuracy results

### Loading a Saved Model
```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('saved_model/cifar10_cnn_model.h5')

# Make predictions
predictions = model.predict(test_images)
```

### Loading Saved Dataset
```python
import numpy as np

# Load saved numpy arrays
x_train = np.load('saved_data/x_train.npy')
y_train = np.load('saved_data/y_train.npy')
x_test = np.load('saved_data/x_test.npy')
y_test = np.load('saved_data/y_test.npy')
```

## Results

### Performance Metrics
- **Training Accuracy:** ~85-90% (after 15 epochs)
- **Validation Accuracy:** ~80-85%
- **Test Accuracy:** ~78-83%

### Training Visualization
The training process generates an accuracy plot (`training_validation_accuracy.png`) showing:
- Training accuracy over epochs
- Validation accuracy over epochs

This helps identify overfitting or underfitting issues.

## Implementation Steps

### Step 1: Data Loading & Preprocessing
- Load CIFAR-10 using `tf.keras.datasets.cifar10.load_data()`
- Normalize pixel values to range [0, 1] by dividing by 255.0
- Convert labels to one-hot encoded format

### Step 2: Model Building
- Define Sequential model with Conv2D, MaxPooling2D, Dense, and Dropout layers
- Use appropriate activation functions (ReLU for hidden layers, Softmax for output)

### Step 3: Model Compilation
- Set optimizer to Adam
- Use categorical cross-entropy loss for multi-class classification
- Track accuracy metric

### Step 4: Model Training
- Train with `model.fit()` for 15 epochs
- Use batch size of 64
- Split 20% of training data for validation

### Step 5: Evaluation & Saving
- Evaluate on test set
- Save model weights in HDF5 format
- Save training history plots
- Export dataset arrays for reproducibility

## Challenges & Solutions

### Challenge 1: Overfitting
**Problem:** Model performs well on training data but poorly on validation/test data.  
**Solution:** Added Dropout layer (0.5) to reduce overfitting and improve generalization.

### Challenge 2: Limited Training Data
**Problem:** 50,000 training images may not be sufficient for complex patterns.  
**Solution:** Consider data augmentation (rotation, flipping, zooming) in future iterations.

### Challenge 3: Computational Resources
**Problem:** Training CNNs can be time-consuming on CPU.  
**Solution:** Utilize GPU acceleration with TensorFlow-GPU or tensorflow-metal on Apple Silicon.

### Challenge 4: Low Resolution Images
**Problem:** 32x32 pixels is relatively low resolution, limiting feature extraction.  
**Solution:** Design deeper CNN architecture with multiple convolutional layers to capture hierarchical features.

### Challenge 5: macOS Installation Issues
**Problem:** NumPy/TensorFlow library loading errors on macOS.  
**Solution:** Use conda environment with proper BLAS libraries (OpenBLAS) and tensorflow-macos for Apple Silicon.

## Future Improvements

1. **Data Augmentation:** Apply random transformations to training images
2. **Transfer Learning:** Use pre-trained models (ResNet, VGG) and fine-tune
3. **Hyperparameter Tuning:** Experiment with learning rates, batch sizes, optimizers
4. **Deeper Architectures:** Add more convolutional layers and residual connections
5. **Regularization Techniques:** Batch normalization, L2 regularization
6. **Ensemble Methods:** Combine predictions from multiple models

## Dependencies
See `requirements.txt` for complete list. Main libraries:
- TensorFlow 2.13+ (or tensorflow-macos for Apple Silicon)
- NumPy 1.24+
- Matplotlib 3.7+

## References
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- TensorFlow CNN Tutorial: https://www.tensorflow.org/tutorials/images/cnn
- Keras Documentation: https://keras.io/api/datasets/cifar10/

## License
This project is available under the MIT License.

## Author
[Your Name]  
[Your Email/GitHub]  
[Date: October 2025]

## Acknowledgments
- Canadian Institute For Advanced Research (CIFAR) for the dataset
- TensorFlow/Keras team for the deep learning framework
- Open-source community for tutorials and documentation

---

**Note:** This project is part of a Deep Learning assignment demonstrating CNN implementation for image classification tasks.
