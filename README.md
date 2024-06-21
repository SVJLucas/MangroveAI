
# MangroveAI

MangroveAI is a deep learning-based approach for mangrove monitoring and conservation using satellite imagery. This repository contains the code and data for the paper "Deep Learning for Mangrove Conservation: Improved Mapping with Mamba". The project aims to enhance mangrove segmentation accuracy by leveraging advanced deep learning models, including convolutional, transformer, and Mamba architectures.

## Overview

Mangroves are vital coastal ecosystems that play a crucial role in environmental health, economic stability, and climate resilience. This project focuses on developing and evaluating state-of-the-art deep learning models for accurate mangrove segmentation from multispectral satellite imagery. The key contributions of this work include:
- Introducing a novel open-source dataset, MagSet-2, incorporating mangrove annotations from the Global Mangrove Watch and satellite images from Sentinel-2.
- Benchmarking six deep learning architectures: U-Net, PAN, MANet, BEiT, SegFormer, and Swin-UMamba.
- Demonstrating the superior performance of the Swin-UMamba model in mangrove segmentation tasks.

## Dataset

### MagSet-2

MagSet-2 is an open-source dataset developed specifically for this project. It integrates mangrove annotations from the Global Mangrove Watch with multispectral satellite images from Sentinel-2 for the year 2020, resulting in more than 10,000 paired images and mangrove locations. The dataset encompasses images from various geographic zones, ensuring a diverse representation of mangrove ecosystems worldwide. This extensive dataset aims to facilitate researchers in training their models to utilize Sentinel-2 imagery for monitoring mangrove areas of environmental protection for years beyond 2020.



<p align="center">
  <img src="https://github.com/SVJLucas/MangroveAI/assets/60625769/778a04e4-45ff-4085-856d-7a3b5d0d2d48" alt="MagSet-2 Dataset" width="700px"/>
</p>
<div align="center">
Sentinel-2 Spectral Display and Vegetation Analysis: Starting from the top left with the RGB bands, followed by the NIR band, Vegetation NIR, and SWIR band in sequence. On the bottom row, from left to right, we have the estimated NDVI, NDWI, NDMI indices, and the targeted Mangrove locations for predictive modeling.
</div>
<br>

Additional perspectives from the dataset are presented, showcasing a diverse array of views from various regions globally. These perspectives highlight the extensive geographical coverage and varied contexts of the dataset, offering a comprehensive representation of mangrove ecosystems across different continents and climatic zones. This diversity underscores the dataset's global relevance and the importance of addressing the unique environmental characteristics present in each region:

<p align="center">
  <img src="https://github.com/SVJLucas/MangroveAI/assets/60625769/02e269b4-aa16-45cd-b0b0-b446281e8d8c" alt="MagSet-2 Dataset (Other views)" width="500px"/>
</p>
<div align="center">
Samples from the MagSet-2 dataset are presented. On the right, the RGB bands are displayed, while on the left, the RGB bands along with the mangrove mask (highlighted in yellow) are shown. The other spectral bands for each sample are not displayed.
</div>
<br>






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

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

