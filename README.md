# ResNet-CIFAR-10

This repository contains the implementation of the ResNet model trained on the CIFAR-10 dataset. The project demonstrates the application of deep residual networks to image classification tasks.

## Overview

ResNet-CIFAR-10 provides a detailed example of using Residual Networks (ResNet) for classifying images from the CIFAR-10 dataset. The implementation showcases the power of deep learning models in handling complex image recognition tasks.

## Features

- **Data Loading and Preprocessing** - (DataReader.py): Functions to load and preprocess CIFAR-10 dataset, including data augmentation techniques like random cropping and horizontal flipping.
- **ResNet Architectures** - (NetWork.py): Implementation of standard and bottleneck ResNet blocks, with projection shortcuts for matching dimensions.
- **Training and Validation** (main.py): Detailed training procedures with dynamic learning rate adjustments and real-time monitoring for best performance.
- **Hyperparameter Tuning** - (main.py): Thorough documentation on tuning experiments, including batch size, ResNet versions, and block sizes, aimed at optimizing model accuracy.
- **Model Evaluation** - (main.py): Test set evaluation to ensure model robustness and generalization.

## Prerequisites

Before running this project, you need to have Python installed along with the following libraries:
- TensorFlow
- NumPy

## Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/tanvu10/ResNet-CIFAR-10.git
cd ResNet-CIFAR-10
```


## Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- NumPy

### Installation

Clone this repository:
```bash
git clone https://github.com/tanvu10/ResNet-CIFAR-10.git
cd ResNet-CIFAR-10
pip install -r requirements.txt
```

### Running the Model
To start training the model:
```bash
python main.py
```

## Documentation
For more details on the architecture and usage, refer to the inline comments and the docs folder