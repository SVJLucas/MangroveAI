
# MangroveAI

MangroveAI is a deep learning-based approach for mangrove monitoring and conservation using satellite imagery. This repository contains the code and data for the paper "Deep Learning for Mangrove Conservation: Improved Mapping with Mamba". The project aims to enhance mangrove segmentation accuracy by leveraging advanced deep learning models, including convolutional, transformer, and Mamba architectures.

## Overview

Mangroves are vital coastal ecosystems that play a crucial role in environmental health, economic stability, and climate resilience. This project focuses on developing and evaluating state-of-the-art deep learning models for accurate mangrove segmentation from multispectral satellite imagery. The key contributions of this work include:
- Introducing a novel open-source dataset, MagSet-2, incorporating mangrove annotations from the Global Mangrove Watch and satellite images from Sentinel-2.
- Benchmarking six deep learning architectures: U-Net, PAN, MANet, BEiT, SegFormer, and Swin-UMamba.
- Demonstrating the superior performance of the Swin-UMamba model in mangrove segmentation tasks.

## Dataset

### MagSet-2
MagSet-2 is an open-source dataset created for this project. It combines mangrove annotations from the Global Mangrove Watch with multispectral satellite images from Sentinel-2. The dataset includes images from various geographic zones, ensuring diverse representation of mangrove ecosystems worldwide.

### Sentinel-2 Imagery
Sentinel-2 satellites provide multispectral imagery crucial for observing mangrove ecosystems. The dataset includes bands in the visible, infrared, and short-wave infrared regions, as well as derived vegetation indices such as NDVI, NDWI, and NDMI.

## Models

The following deep learning models were evaluated:
1. **U-Net**: A convolutional neural network with a symmetrical encoder-decoder architecture.
2. **PAN**: Pyramid Attention Network leveraging pyramid pooling and attention mechanisms.
3. **MANet**: Multi-scale Attention Network incorporating dense connections.
4. **BEiT**: Transformer-based model using self-supervised learning for image representation.
5. **SegFormer**: Transformer-based model with a hierarchical encoder and lightweight decoder.
6. **Swin-UMamba**: A Mamba-based architecture using the Swin-Transformer for enhanced performance.

## Results

The table below summarizes the performance of the models on the MagSet-2 dataset:

| Method       | # Parameters (M) | IoU (%) | Accuracy (%) | F1-score (%) | Loss  |
|--------------|-------------------|---------|--------------|--------------|-------|
| U-Net        | 32.54             | 61.76   | 78.59        | 76.32        | 0.47  |
| PAN          | 34.79             | 64.44   | 81.16        | 78.32        | 0.41  |
| MANet        | 33.38             | 71.75   | 85.80        | 83.51        | 0.34  |
| BEiT         | 33.59             | 70.78   | 85.66        | 82.87        | 0.48  |
| SegFormer    | 34.63             | 72.32   | 86.13        | 83.91        | 0.42  |
| Swin-UMamba  | 32.35             | 72.87   | 86.64        | 84.27        | 0.31  |

Swin-UMamba demonstrated the best overall performance across all metrics, highlighting its effectiveness in mangrove segmentation tasks.

## Usage

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/MangroveAI.git
cd MangroveAI
