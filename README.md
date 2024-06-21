
# MangroveAI

MangroveAI is a deep learning-based approach for mangrove monitoring and conservation using satellite imagery. This repository contains the code and data for the paper **"A Deep Learning-Based Approach for Mangrove Monitoring"**. This work aims to enhance mangrove segmentation accuracy by leveraging advanced deep learning models, including convolutional, transformer, and Mamba architectures.

# For Reviewers
Here we release the training codes for the models for peer review. **For the dataset, we are providing only a few samples due to storage constraints related to its size.** The remaining samples **will be made available to the public (open-source) after peer review** on platforms appropriate for supporting the dataset size.


## Overview

Mangroves are vital coastal ecosystems that play a crucial role in environmental health, economic stability, and climate resilience. This work focuses on developing and evaluating state-of-the-art deep learning models for accurate mangrove segmentation from multispectral satellite imagery. The key contributions of this work include:
- Introducing a novel open-source dataset, MagSet-2, incorporating mangrove annotations from the Global Mangrove Watch and satellite images from Sentinel-2.
- Benchmarking six deep learning architectures: U-Net, PAN, MANet, BEiT, SegFormer, and Swin-UMamba.
- Demonstrating the superior performance of the Swin-UMamba model in mangrove segmentation tasks.

## Dataset MagSet-2


MagSet-2 is an open-source dataset we developed specifically for this work. It integrates mangrove annotations from the _Global Mangrove Watch_ with multispectral satellite images from _Sentinel-2_ for the year 2020, resulting in more than 10,000 paired images and mangrove locations. The dataset encompasses images from various geographic zones, ensuring a diverse representation of mangrove ecosystems worldwide. This extensive dataset aims to facilitate researchers in training their models to utilize _Sentinel-2_ imagery for monitoring mangrove areas of environmental protection for years beyond 2020.

<p align="center">
  <img src="images/data_locations.png" alt="Sample Locations on World Map " width="600px"/>
</p>
<div align="center">
Mangrove Position (neon blue) and the different Mangrove Zones (green) Dataset based on the Global Mangrove Watch (GMW) v3.2020. Each Neon Blue Point represents a position of a sample from the MagSet-2 dataset.
</div>
<br>

Our dataset includes various bands from the electromagnetic spectrum, obtained from Sentinel-2 imagery. It features RGB (Red, Green, Blue) bands, the Near-Infrared (NIR) band, Vegetation NIR band, and Short-Wave Infrared (SWIR) band. Additionally, it includes estimated vegetation indices such as NDVI, NDWI, and NDMI, alongside targeted mangrove locations. These diverse spectral bands and indices enhance predictive modeling, enabling precise and detailed mangrove segmentation for effective monitoring and conservation:

<p align="center">
  <img src="images/sample_from_magset2.png" alt="MagSet-2 Dataset" width="700px"/>
</p>
<div align="center">
Sentinel-2 Spectral Display and Vegetation Analysis: Starting from the top left with the RGB bands, followed by the NIR band, Vegetation NIR, and SWIR band in sequence. On the bottom row, from left to right, we have the estimated NDVI, NDWI, NDMI indices, and the targeted Mangrove locations for predictive modeling.
</div>
<br>

Additional perspectives from the dataset are presented, showcasing a diverse array of views from various regions globally. These perspectives highlight the extensive geographical coverage and varied contexts of the dataset, offering a comprehensive representation of mangrove ecosystems across different continents and climatic zones. This diversity underscores the dataset's global relevance and the importance of addressing the unique environmental characteristics present in each region:

<p align="center">
  <img src="images/mangrove_masks.png" alt="MagSet-2 Dataset (Other views)" width="500px"/>
</p>
<div align="center">
Samples from the MagSet-2 dataset are presented. On the right, the RGB bands are displayed, while on the left, the RGB bands along with the mangrove mask (highlighted in yellow) are shown. The other spectral bands for each sample are not displayed.
</div>
<br>



## Models

The following deep learning models were evaluated:

### Convolutional-based Architectures 
- **U-Net**: A convolutional neural network with a symmetrical encoder-decoder architecture.
- **PAN**: Pyramid Attention Network leveraging pyramid pooling and attention mechanisms.
- **MANet**: Multi-scale Attention Network incorporating dense connections.
### Transformer-based Architectures 
- **BEiT**: Transformer-based model using self-supervised learning for image representation.
- **SegFormer**: Transformer-based model with a hierarchical encoder and lightweight decoder.
### Mamba-based Architectures 
- **Swin-UMamba**: A Mamba-based architecture using the Swin-Transformer for enhanced performance.

## Training Pipeline

<p align="center">
  <img src="images/papers_pipeline.png" alt="Pipeline" width="800px"/>
</p>
<div align="center">
A flowchart representing the steps of the satellite image processing pipeline for the prediction of mangrove locations. The process begins with the acquisition of Sentinel-2 imagery from the Copernicus dataset, followed by zone definition, annotation assignment, data filtering, data augmentation, and image normalization. These pre-processed images are then fed into a predictive model, the output of which is compared against the Global Mangrove Watch v3.2020 dataset to validate the model's predictions.
</div>
<br>



## Results

We assessed the performance of each model using established image segmentation evaluation metrics, complemented by a qualitative analysis of the results. The key metrics included:

- **Intersection over Union (IoU)**: Measures the overlap between the predicted masks and the actual ground truth, crucial for evaluating segmentation precision.
- **Accuracy**: Gauges the proportion of correctly classified samples.
- **F1-score**: Balances precision and recall.
- **Loss (Binary Cross-Entropy)**: Helps optimize the training process.


| Method       | # Parameters (M) | IoU (%) | Accuracy (%) | F1-score (%) | Loss  |
|--------------|-------------------|---------|--------------|--------------|-------|
| U-Net        | 32.54             | 61.76   | 78.59        | 76.32        | 0.47  |
| PAN          | 34.79             | 64.44   | 81.16        | 78.32        | 0.41  |
| MANet        | 33.38             | 71.75   | 85.80        | 83.51        | 0.34  |
| BEiT         | 33.59             | 70.78   | 85.66        | 82.87        | 0.48  |
| SegFormer    | 34.63             | 72.32   | 86.13        | 83.91        | 0.42  |
| Swin-UMamba  | 32.35             | **72.87** | **86.64**    | **84.27**    | **0.31** |


The performance of the selected deep learning models for mangrove segmentation on Sentinel-2 satellite imagery is presented in the table below. The models are categorized into three architectural groups:

- **Convolutional models**: U-Net, PAN, and MANet. U-Net had the lowest number of parameters (32.54 million) and the lowest IoU score (61.76%). MANet achieved the lowest test loss (0.34) among the convolutional models, while PAN had a middle-ground performance.
- **Transformer models**: BEiT and SegFormer. These showed improved performance compared to convolutional models, with SegFormer achieving the highest IoU score (72.32%) within this group.
- **Mamba model**: Swin-UMamba outperformed all other models across all metrics, achieving the highest IoU (72.87%), accuracy (86.64%), F1-score (84.27%), and the lowest loss (0.31).




<p align="center">
  <img src="images/evaluation.png" alt="Evaluation" width="800px"/>
</p>
<div align="center">
Comparative performance on Sentinel-2 test, using Training Set Loss (left), Test Set F1 Score (center), and Test Set Intersection over Union (IoU) (right). Each line represents a model: U-Net (neon blue), PAN (red), MANet (black), BEiT (green), SegFormer (yellow), and Swin-UMamba (dark blue) trained over 100 epochs. Lower loss values, higher F1 and IoU values indicate better performance. Swin-UMamba consistently shows superior performance over all metrics.
</div>
<br>


<p align="center">
  <img src="images/model_results.png" alt="Comparative visual segmentation results of mangrove areas" width="1000px"/>
</p>
<div align="center">
Comparative visual segmentation results of mangrove areas. The first column shows the original satellite images, the second column depicts the ground truth segmentation, and the subsequent columns display the segmentation results from U-Net, PAN, MANet, BEiT, SegFormer, and Swin-UMamba models.
</div>
<br>


# License
This work is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

