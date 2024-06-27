import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def plot_augmented_images(test_image: np.ndarray, true_mask: np.ndarray, image_index, save_bands) -> None:
    """
    Plot augmented RGB, infrared images, NDVI, and mangrove mask side by side.

    Args:
        test_image: A numpy array representing the input image.
        true_mask: A numpy array representing the mangrove mask.

    Returns:
        None. Displays the images in a 2x2 grid.
    """

    # Split the image into bands
    image_rgb = test_image[:,:, [2, 1, 0]]
    image_nir = test_image[:,:, 3]
    image_nir_veg = test_image[:,:, 4]
    image_swir = test_image[:,:, 5]
    ndvi = test_image[:,:, 6]
    ndwi = test_image[:,:, 7]
    ndmi = test_image[:,:, 8]

    # Display bands and true mangrove annotations
    plt.figure(figsize=(15, 10))

    # Plot RGB image
    plt.subplot(2, 4, 1)
    clip_range = (0, 1)
    plt.imshow(2*image_rgb)
    # plt.imshow(np.where(1.3*augmented_rgb>1,1,1.3*augmented_rgb))
    plt.title("Satellite RGB")
    plt.axis('off')

    # Plot NIR image with 'hot' colormap
    plt.subplot(2, 4, 2)
    plt.imshow(image_nir, cmap='hot')
    plt.title("Satellite NIR")
    plt.axis('off')

    # Plot Vegetation NIR image with 'Wistia' colormap
    plt.subplot(2, 4, 3)
    plt.imshow(image_nir_veg, cmap='Wistia')
    plt.title("Satellite Vegetation NIR")
    plt.axis('off')

    # Plot SWIR image with 'cool' colormap
    plt.subplot(2, 4, 4)
    plt.imshow(image_swir, cmap='cool')
    plt.title("Satellite SWIR")
    plt.axis('off')

    # Plot NDVI with 'viridis' colormap
    plt.subplot(2, 4, 5)
    plt.imshow(ndvi, cmap='viridis')
    plt.title("Estimated NDVI")
    plt.axis('off')

    # Plot NDMI with 'summer' colormap
    plt.subplot(2, 4, 6)
    plt.imshow(ndwi, cmap='summer')
    plt.title("Estimated NDWI")
    plt.axis('off')

    # Plot NDVI with 'inferno' colormap
    plt.subplot(2, 4, 7)
    plt.imshow(ndmi, cmap='inferno')
    plt.title("Estimated NDMI")
    plt.axis('off')

    # Plot mangrove mask with 'cividis' colormap
    plt.subplot(2, 4, 8)
    plt.imshow(true_mask, cmap= 'cividis')
    plt.title("Mangrove Locations from GMW")
    plt.axis('off')

    plt.tight_layout()

    # Save the plot
    file_path = os.path.join(save_bands, f'plot_comparison_{image_index}.png')
    plt.savefig(file_path)
    print(image_index)
    plt.show()

def plot_segmentation_results(batch, device, model_save_path, best_model, best_model_filename, save_comparisons):
    """
    Plot segmentation results.

    Args:
        batch: The batch of images and true masks.
        model_save_path (str): The path to the folder where the model is saved.
        best_model_filename (str): The filename of the best model.

    Returns:
        None. Displays the segmentation results.
    """

    best_model_path = os.path.join(model_save_path, best_model_filename)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    with torch.no_grad():
        best_model.eval()
        logits = best_model(batch[0].to(device))

    # Counter variable to track iterations
    image_index = 0

    for image, true_mask, logit in zip(batch[0], batch[1], logits):

        # Predicted mask
        base_mask = logit.cpu().detach().numpy()
        pred_mask = np.where(base_mask >= 0.5, 1, 0).squeeze()  # just squeeze classes dim, because we have only one class

        # Brighter image RGB
        clip_range = (0, 1)
        test_image = image.permute(1, 2, 0).cpu().detach().numpy()  # convert CHW -> HWC with permute (transpose for tensors)
        image_rgb = test_image[:, :, [2, 1, 0]]  # take RGB bands in the order BGR
        brighter_image_rgb = 3 * image_rgb

        # True mask
        true_mask = true_mask[0].cpu().detach().numpy().astype('int')

        # Compute evaluation metrics
        flat_true_mask, flat_pred_mask = true_mask.flatten(), pred_mask.flatten()
        accuracy = accuracy_score(flat_true_mask, flat_pred_mask)
        precision = precision_score(flat_true_mask, flat_pred_mask)
        recall = recall_score(flat_true_mask, flat_pred_mask)
        f1 = f1_score(flat_true_mask, flat_pred_mask)
        iou = jaccard_score(flat_true_mask, flat_pred_mask)

        np.save(os.path.join(save_comparisons, f"image_{image_index}.npy"), test_image)
        np.save(os.path.join(save_comparisons, f"true_mask_{image_index}.npy"), true_mask)
        np.save(os.path.join(save_comparisons, f"logit_{image_index}.npy"), base_mask)

        # Custom colormaps
        cmap_model = mcolors.ListedColormap(['none', 'yellow'])
        cmap_mangrove = mcolors.ListedColormap(['none', 'blue'])

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

        # Plot 1 - Original image
        axs[0].imshow(brighter_image_rgb)
        axs[0].set_title('Satellite observation from Sentinel-2')
        axs[0].axis('off')  # Turn off axis

        # Plot 2 - Model Output
        axs[1].imshow(brighter_image_rgb)  # Base image
        axs[1].imshow(pred_mask, cmap=cmap_model, alpha=0.5)  # Overlay
        axs[1].set_title(
            f'Mangrove Position (Model) \n Acc: {accuracy:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}, IoU: {iou:.2f}')
        axs[1].axis('off')  # Turn off axis

        # Plot 3 - Mangrove Position
        axs[2].imshow(brighter_image_rgb)  # Base image
        axs[2].imshow(true_mask, cmap=cmap_mangrove, alpha=0.5)  # Overlay
        axs[2].set_title('Mangrove Position (GWM)')
        axs[2].axis('off')  # Turn off axis

        file_path = os.path.join(save_comparisons, f'plot_comparison_{image_index}.png')
        plt.savefig(file_path)

        plt.show()

        # Increment index
        image_index += 1

