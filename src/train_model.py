import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
from tqdm import tqdm
import time
import datetime
import os

def train_final_model(model, total_training_loader, total_test_loader, n_epochs, optimizer, scheduler, criterion, device, model_save_path_epochs, save_interval):
    """
    Trains a PyTorch model.

    Args:
    - model (nn.Module): the PyTorch model to be trained
    - total_training_loader (DataLoader): DataLoader containing the training set
    - total_test_loader (DataLoader): DataLoader containing the test set
    - n_epochs (int): number of training epochs
    - optimizer (optim.Optimizer): the optimizer to use for training
    - scheduler (optim.lr_scheduler): the scheduler used for training
    - criterion (nn.Module): the loss function to use for training
    - device (str): device to run the model on (e.g. 'cpu' or 'cuda')
    - model_save_path_epochs (str): path to the folder where the model will be saved
    - save_interval (int): save the model every save_interval epochs

    Returns:
    - mean_loss_train (list): mean loss per epoch for the training set
    - mean_f1_train (list): mean F1 score per epoch for the training set
    - mean_iou_train (list): mean IOU per epoch for the training set
    - mean_accuracy_train (list): mean accuracy per epoch for the training set
    - mean_loss_test (list): mean loss per epoch for the test set
    - mean_f1_test (list): mean F1 score per epoch for the test set
    - mean_iou_test (list): mean IOU per epoch for the test set
    - mean_accuracy_test (list): mean accuracy per epoch for the test set
    - elapsed_time (float): total training time
    - best_model_filename (str): filename of the best model
    - best_mean_iou_test (float): best IOU score achieved on the test set
    """

    # Initialize lists to keep track of metrics
    mean_loss_train = []
    mean_f1_train = []
    mean_iou_train = []
    mean_accuracy_train = []

    mean_loss_test = []
    mean_f1_test = []
    mean_iou_test = []
    mean_accuracy_test = []

    best_mean_iou_test = -np.inf
    best_model_filename = None

    start_time = time.time()

    # Loop over epochs
    for it in range(1, n_epochs + 1):

        print(f"EPOCH {it}")

        print("Training...")

        # Initialize lists to keep track of train metrics for this epoch
        train_loss = []
        train_f1 = []
        train_iou = []
        train_accuracy = []

        # Set model to train mode
        model.train()

        # Loop over training data
        for images, targets in tqdm(total_training_loader):

            # Move data to device
            images, targets = images.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Append loss to train_loss list
            train_loss.append(loss.item())

            # Calculate F1 score
            preds_binary = (outputs > 0.5).float()
            f1 = f1_score(targets.cpu().numpy().flatten(), preds_binary.cpu().numpy().flatten())

            train_f1.append(f1)

            # Calculate IOU score
            iou = jaccard_score(targets.cpu().numpy().flatten(), preds_binary.cpu().numpy().flatten())
            train_iou.append(iou)

            # Calculate accuracy
            accuracy = accuracy_score(targets.cpu().numpy().flatten(), preds_binary.cpu().numpy().flatten())
            train_accuracy.append(accuracy)

        print("Testing...")
        # Initialize lists to keep track of test metrics for this epoch
        test_loss = []
        test_f1 = []
        test_iou = []
        test_accuracy = []

        # Set model to evaluation mode
        model.eval()

        # Turn off gradients for evaluation
        with torch.no_grad():

            # Loop over test data
            for images, targets in tqdm(total_test_loader):

                # Move data to device
                images, targets = images.to(device), targets.to(device)

                # Forward pass
                outputs = model(images)

                # Compute loss
                loss = criterion(outputs, targets)

                # Append loss to test_loss list
                test_loss.append(loss.item())

                # Calculate F1 score
                preds_binary_test = (outputs > 0.5).float()
                f1 = f1_score(targets.cpu().numpy().flatten(), preds_binary_test.cpu().numpy().flatten())
                test_f1.append(f1)

                # Calculate IOU score
                iou = jaccard_score(targets.cpu().numpy().flatten(), preds_binary_test.cpu().numpy().flatten())
                test_iou.append(iou)

                # Calculate accuracy
                accuracy = accuracy_score(targets.cpu().numpy().flatten(), preds_binary_test.cpu().numpy().flatten())
                test_accuracy.append(accuracy)

        # Step the scheduler
        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']

        # Append the mean train metrics for this epoch to the lists
        mean_loss_train.append(np.mean(train_loss))
        mean_f1_train.append(np.mean(train_f1))
        mean_iou_train.append(np.mean(train_iou))
        mean_accuracy_train.append(np.mean(train_accuracy))

        # Append the mean test metrics for this epoch to the lists
        mean_loss_test.append(np.mean(test_loss))
        mean_f1_test.append(np.mean(test_f1))
        mean_iou_test.append(np.mean(test_iou))
        mean_accuracy_test.append(np.mean(test_accuracy))

        # Print epoch metrics
        print(f'Epoch {it}/{n_epochs}, Train Loss: {mean_loss_train[-1]:.4f}, Train F1: {mean_f1_train[-1]:.4f}, Train IOU: {mean_iou_train[-1]:.4f}, Train Accuracy: {mean_accuracy_train[-1]:.4f}, Test Loss: {mean_loss_test[-1]:.4f}, Test F1: {mean_f1_test[-1]:.4f}, Test IOU: {mean_iou_test[-1]:.4f}, Test Accuracy: {mean_accuracy_test[-1]:.4f}, lr: {lr}, Elapsed time: {((time.time() - start_time) / 60):.2f} minutes')

        # Saving the model if mean_iou_test has improved
        current_mean_iou_test = mean_iou_test[-1]
        if current_mean_iou_test > best_mean_iou_test:
            print(f"Saving model...")
            best_mean_iou_test = current_mean_iou_test
            best_model_filename = f'mangrove_epoch_{it}_iou_test_{current_mean_iou_test:.4f}_date_{datetime.datetime.now().strftime("%d%m%Y-%H%M%S")}.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path_epochs, best_model_filename))

    # Record the end time
    end_time = time.time()

    # Compute the elapsed time in seconds
    elapsed_time = end_time - start_time     

    return mean_loss_train, mean_f1_train, mean_iou_train, mean_accuracy_train, mean_loss_test, mean_f1_test, mean_iou_test, mean_accuracy_test, elapsed_time, best_model_filename, best_mean_iou_test

def save_metrics_to_file(metrics, model_save_path_metrics):
    """
    Save metrics to a file.

    Args:
    - metrics (dict): Dictionary containing metrics to be saved
    - model_save_path(str): Path to the file where metrics will be saved
    """

    with open(model_save_path_metrics, 'w') as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    print("Lists data saved successfully.")
