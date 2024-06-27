import os
import glob
import random
from typing import List
import numpy as np
from skimage.io import imread
import albumentations as A
import torch
from torch.utils.data import Dataset

def get_all_file_paths(folder_path):
    # Use glob to get all file paths in the folder and its subfolders
    file_paths = glob.glob(os.path.join(folder_path, '**'), recursive=True)
    # Filter out directories from the list
    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]
    return file_paths

def get_all_file_paths_by_zone(folder_path, n_zones):
    file_paths = glob.glob(os.path.join(folder_path, '**'), recursive=True)
    file_paths_by_zone = {i:[] for i in range(n_zones)}
    for file_path in file_paths:
        if os.path.isfile(file_path):
            zone_id = int(file_path.split("/")[-1][0])
            file_paths_by_zone[zone_id].append(file_path)
    return file_paths_by_zone

class MangroveSegmentationDataset(Dataset):
    def __init__(self, paths:List, bands_to_keep = None, use_augmentation=False):
        """
        Initialize the dataset for Mangrove segmentation.
        """
        self.paths = paths
        random.shuffle(self.paths)  # Shuffle the entire list of paths
        self.bands_to_keep = bands_to_keep
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Get the image and mask for a given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: Tuple containing the image and mask tensors.
        """
        # Construct file paths for the image and mask
        image_path, mask_path = self.paths[idx]

        # Load image and mask
        try:
            # only keep requested bands
            image = imread(image_path)[:,:,self.bands_to_keep].astype('float32') # Ensure the dtype for augmentations
            mask = np.load(mask_path)
        except Exception as e:
            print(f"Error loading image/mask at index {idx}: {e}")
            return None, None

        # Apply transforms
        if self.use_augmentation == True:
            augmentation_transform = A.Compose([A.VerticalFlip(p=0.5),
                               A.HorizontalFlip(p=0.5),
                               A.RandomRotate90(p=0.5),
                               A.RandomResizedCrop(128, 128, p=1),
                               ])
            sample = {"image" : image, "mask": mask}
            transformed_sample = augmentation_transform(**sample)
            image, mask = transformed_sample["image"].copy(), transformed_sample["mask"].copy()

       # Normalize image
        post_processed_image = (image-image.min())/(image.max()-image.min() + 1e-8) # Normalize image channel by channel

        if self.use_augmentation == True:
            # Add Gaussian Noise (to the normalized image because values need to be between 0 and 1)
            gaussian_transform = A.GaussNoise(var_limit=(0.002), p=0.5)  # Adjust var_limit for the strength of noise
            post_processed_image = gaussian_transform(image=post_processed_image)["image"].copy()

         # Convert to PyTorch tensors
        image_tensor = torch.as_tensor(post_processed_image.transpose(2, 0, 1), dtype=torch.float) # Images have to be in Channel*Height*Width (CHW) format but we read them as HWC
        mask_tensor = torch.as_tensor(np.expand_dims(mask, axis=0), dtype=torch.float) # Add a dimension to have masks in the CHW (C=1) format and not HW

        return image_tensor, mask_tensor

def get_train_test_paths_by_zone(dataset_dir, train_test_split, n_samples_per_zone, n_zones):

    # Train/Test split on shuffled indexes in the range(len(image_paths))
    full_paths_train = []
    full_paths_test = []

    if n_samples_per_zone:
        # Only keep n_samples_per_zone samples for the benchmarking
        image_paths_by_zone = get_all_file_paths_by_zone(os.path.join(dataset_dir, "satellite-images/"), n_zones)
        mask_paths_by_zone = get_all_file_paths_by_zone(os.path.join(dataset_dir, "masks/"), n_zones)

        for zone_id in range(n_zones):
            image_paths, mask_paths = image_paths_by_zone[zone_id], mask_paths_by_zone[zone_id]
            assert len(image_paths) == len(mask_paths), f"Unequal number of images and masks for zone {zone_id}"
            sample_indexes = list(range(len(image_paths)))
            random.shuffle(sample_indexes)
            selected_indices = random.sample(sample_indexes, n_samples_per_zone)
            n_train_samples = int(n_samples_per_zone * train_test_split)
            train_indices, test_indices = selected_indices[:n_train_samples], selected_indices[n_train_samples:]
            full_paths_train += [(image_paths[idx], mask_paths[idx]) for idx in train_indices]
            full_paths_test += [(image_paths[idx], mask_paths[idx]) for idx in test_indices]

    return full_paths_train, full_paths_test


def get_train_test_paths(dataset_dir, train_test_split):

    image_paths = sorted(get_all_file_paths(os.path.join(dataset_dir, "satellite-images/")))
    mask_paths = sorted(get_all_file_paths(os.path.join(dataset_dir, "masks/")))
    assert len(image_paths) == len(mask_paths), f"Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) should be equal"
    n_samples = len(image_paths)
    print(f"Total number of selected samples = {n_samples}")
    n_train_samples = int(n_samples * train_test_split)
    samples_indexes = list(range(n_samples))
    random.shuffle(samples_indexes)
    train_indices, test_indices = samples_indexes[:n_train_samples], samples_indexes[n_train_samples:]
    assert (len(train_indices + test_indices)) == n_samples, f"Sum of train ({len(train_indices)}) and test ({len(test_indices)}) should be equal to {n_samples}"

    full_paths_train = [(image_paths[idx], mask_paths[idx]) for idx in train_indices]
    full_paths_test = [(image_paths[idx], mask_paths[idx]) for idx in test_indices]

    return full_paths_train, full_paths_test
