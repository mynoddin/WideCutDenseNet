# WideCutDenseNet: A Novel CNN Architecture for CIFAR-10 Classification

This repository presents **WideCutDenseNet**, a custom Convolutional Neural Network (CNN) architecture designed for high-accuracy image classification on the CIFAR-10 dataset. The model combines concepts from DenseNet, Wide ResNet, and attention mechanisms to deliver competitive performance with fewer parameters.

## 🧠 Project Highlights

- Hybrid CNN architecture: **Wide layers + Dense connections + Attention modules**
- Designed for efficiency and accuracy on CIFAR-10
- Integrates modern deep learning best practices like BatchNorm, Dropout, and Learning Rate Scheduling
- Achieves >97% accuracy with a relatively small model size

## 📁 Project Structure

```
research/
│
├── main.ipynb               # Main Jupyter notebook with end-to-end pipeline
├── data/                    # CIFAR-10 data (downloaded using torchvision or keras)
├── models/                  # Saved model checkpoints
├── utils/                   # Utility functions (plotting, augmentation, etc.)
├── results/                 # Plots, accuracy/loss curves, and model comparisons
└── README.md                # This file
```

## 📊 Dataset

- **CIFAR-10**: 60,000 32×32 color images in 10 classes
  - 50,000 training images
  - 10,000 testing images

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/WideCutDenseNet.git
   cd WideCutDenseNet
   ```

2. Set up and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook research/main.ipynb
   ```

## 🚀 Training & Evaluation Pipeline

The training pipeline includes:

- Data preprocessing with normalization and augmentation
- Custom model architecture (WideCutDenseNet)
- Training with cross-entropy loss, Adam optimizer, and LR scheduler
- Evaluation using accuracy, confusion matrix, and loss curves

## 📈 Results

| Metric        | Value        |
|---------------|--------------|
| Accuracy      | >97%         |
| Loss          | ↓ Steadily   |
| Parameters    | Fewer than ResNet-50 |
| Performance   | On par with deeper networks |

Plots and model comparisons are available in the `results/` directory.

## 🧠 Model Architecture

WideCutDenseNet combines:

- **Wide Convolutions** for large receptive fields
- **Dense Connections** for efficient feature reuse
- **SE-Attention Modules** for channel recalibration

*(Refer to `model_architecture.png` in the `images/` directory.)*

## 📌 Requirements

- Python 3.8+
- TensorFlow or PyTorch (depending on implementation)
- NumPy, Matplotlib, Scikit-learn
- Jupyter Notebook

## 📚 Reference

If you use this project or draw inspiration from it, please consider citing or referencing:

> _Md Mynoddin et al., "WideCutDenseNet: Novel CNN Architecture for CIFAR-10", 2025_

## 🙌 Acknowledgements

- CIFAR-10 dataset by Alex Krizhevsky
- DenseNet: [Gao Huang et al., CVPR 2017](https://arxiv.org/abs/1608.06993)
- Squeeze-and-Excitation Networks: [Hu et al., CVPR 2018](https://arxiv.org/abs/1709.01507)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
