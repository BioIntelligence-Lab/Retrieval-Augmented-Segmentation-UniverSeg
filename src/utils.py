import pathlib
from typing import Tuple
import os
import einops
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from PIL import Image
from torchvision import transforms
from pylab import *

def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = Image.open(path)
    img = img.resize(size, resample=Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = Image.open(path)
    seg = seg.resize(size, resample=Image.NEAREST)
    seg = np.array(seg)
    seg = np.stack([seg == 0, seg == 128, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*.bmp")):
        img = process_img(file, size=size)
        seg_file = file.with_suffix(".png")
        seg = process_seg(seg_file, size=size)
        data.append((img / 255.0, seg))
    return data
    

def visualize_tensors(tensors, col_wrap=8, col_names=None, title=None):
    M = len(tensors)
    N = len(next(iter(tensors.values())))

    cols = col_wrap
    rows = math.ceil(N/cols) * M

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d*cols, d*rows))
    if rows == 1:
      axes = axes.reshape(1, cols)

    for g, (grp, tensors) in enumerate(tensors.items()):
        for k, tensor in enumerate(tensors):
            col = k % cols
            row = g + M*(k//cols)
            x = tensor.detach().cpu().numpy().squeeze()
            ax = axes[row,col]
            if len(x.shape) == 2:
                ax.imshow(x,vmin=0, vmax=1, cmap='gray')
            else:
                ax.imshow(einops.rearrange(x,'C H W -> H W C'))
            if col == 0:
                ax.set_ylabel(grp, fontsize=16)
            if col_names is not None and row == 0:
                ax.set_title(col_names[col])

    for i in range(rows):
        for j in range(cols):
            ax = axes[i,j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()

def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred
    y_true = y_true
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()


@torch.no_grad()
def inference(model, image, label, support_images, support_labels,device):
    image, label = image.to(device), label.to(device)
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    # inference
    with torch.inference_mode():
        logits = model(image[None],support_images[None],support_labels[None])[0].detach() # outputs are logits
    

    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)
    del logits
    torch.cuda.empty_cache()
    #  score
    score = dice_score(hard_pred, label)

    # return a dictionary of all relevant variables
    return {'Image': image,
            'Soft Prediction': soft_pred,
            'Prediction': hard_pred,
            'Ground Truth': label,
            'score': score}


def plot_tensor(arr1):
    num_rows = 4  # You can adjust this based on your preference
    num_cols = 8
    images_tensor = arr1 
# Create a figure and axes for the grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6))

# Loop through each image in the tensor and plot in the grid
    for i in range(images_tensor.size(0)):
        row_idx = i // num_cols
        col_idx = i % num_cols

    # Convert the tensor to a numpy array
        image_array = images_tensor[i, 0].numpy()

    # Plot the image in the corresponding grid cell
        axs[row_idx, col_idx].imshow(image_array, cmap='gray')  # Assuming it's a grayscale image
        axs[row_idx, col_idx].set_title(f"Image {i+1}")
        axs[row_idx, col_idx].axis('off')
 
# Adjust layout to prevent clipping
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def inference_mix_prec(model, image, label, support_images, support_labels,device):
    image, label = image.to(device), label.to(device)
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    # inference
    with torch.inference_mode(),torch.autocast(device,dtype=torch.bfloat16):
        logits = model(image[None],support_images[None],support_labels[None])[0].detach().float() # outputs are logits
    

    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)
    del logits
    torch.cuda.empty_cache()
    #  score
    score = dice_score(hard_pred, label)

    # return a dictionary of all relevant variables
    return {'Image': image,
            'Soft Prediction': soft_pred,
            'Prediction': hard_pred,
            'Ground Truth': label,
            'score': score}

