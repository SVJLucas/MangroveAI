import matplotlib.pyplot as plt
import os

def plot_train_val_loss(train_losses: list, val_losses: list):
    """
    Plot training and validation losses.

    Args:
        train_losses (list): A list of training losses.
        val_losses (list): A list of validation losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log10(Loss)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_val_iou(train_iou: list, test_iou: list):
    """
    Plot training and validation IOU scores.

    Args:
        train_f1 (list): A list of training IOU scores.
        val_f1 (list): A list of validation IOU scores.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_iou, label='Training IOU Score')
    plt.plot(test_iou, label='Validation IOU Score')
    plt.title('Training vs Validation IOU Score')
    plt.xlabel('Epochs')
    plt.ylabel('IOU Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_val_f1(train_f1: list, test_f1: list):
    """
    Plot training and validation f1 scores.

    Args:
        train_f1 (list): A list of training f1 scores.
        val_f1 (list): A list of validation f1 scores.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_f1, label='Training f1 Score')
    plt.plot(test_f1, label='Validation f1 Score')
    plt.title('Training vs Validation f1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_val_acc(train_acc: list, test_acc: list):
    """
    Plot training and validation acc scores.

    Args:
        train_acc (list): A list of training acc scores.
        val_acc (list): A list of validation acc scores.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

