import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from imageio import imread

def get_all_file_paths(folder_path):
    """
    Get all file paths in the folder and its subfolders.
    """
    file_paths = glob.glob(os.path.join(folder_path, '**'), recursive=True)
    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]
    return file_paths

def get_all_file_paths_by_zone(folder_path, n_zones):
    """
    Get all file paths in the folder and its subfolders grouped by zone.
    """
    file_paths = glob.glob(os.path.join(folder_path, '**'), recursive=True)
    file_paths_by_zone = {i:[] for i in range(n_zones)}
    for file_path in file_paths:
        if os.path.isfile(file_path):
            zone_id = int(file_path.split("/")[-1][0])
            file_paths_by_zone[zone_id].append(file_path)
    return file_paths_by_zone

def display_samples(images, masks, nb_samples: list, palette=mcolors.ListedColormap(['none', 'red'])) -> None:
    """
    Display samples of images and their corresponding masks.
    """
    indices = random.sample(range(0, len(images)), nb_samples)
    fig, axs = plt.subplots(nrows=nb_samples, ncols=3, figsize=(20, nb_samples * 6))
    fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')
    for u, idx in enumerate(indices):
        rgb_im = imread(images[idx])[:, :, [2, 1, 0]]
        mk = np.load(masks[idx])
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0]
        ax0.imshow(np.clip(rgb_im * 5 / 1e4, 0, 1))
        ax0.axis('off')
        ax1 = axs[u][1]
        ax1.imshow(mk, cmap=palette, interpolation='nearest')
        ax1.axis('off')
        ax2 = axs[u][2]
        ax2.imshow(np.clip(rgb_im * 5 / 1e4, 0, 1))
        ax2.imshow(mk, cmap=palette, interpolation='nearest', alpha=0.3)
        ax2.axis('off')
        if u == 0:
            ax0.set_title('RGB Image', size=16, fontweight="bold", c='w')
            ax1.set_title('Ground Truth Mask', size=16, fontweight="bold", c='w')
            ax2.set_title('Overlay Image & Mask', size=16, fontweight="bold", c='w')