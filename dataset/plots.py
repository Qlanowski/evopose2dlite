#%%
import numpy as np
import math
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


NAMES = [
    "Nose", 
    "Left eye", 
    "Right eye", 
    "Left ear",
    "Right ear",
    "Left shoulder",
    "Right shoulder",
    "Left elbow",
    "Right elbow",
    "Left wrist",
    "Right wrist",
    "Left hip",
    "Right hip",
    "Left knee",
    "Right knee",
    "Left ankle",
    "Right ankle",
    "Left big toe",
    "Left small toe",
    "Left heel",
    "Right big toe",
    "Right small toe",
    "Right heel",
]

def to_image(img):
    img2 = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))
    img2[:, :, 0] = img  # same value in each channel
    img2[:, :, 1] = img
    img2[:, :, 2] = img
    return img2

def plot_heatmap(ax, img, hm, idx):
    ax.axis('off')
    h_img = tf.image.resize(to_image(hm), (img.shape[0], img.shape[1]))
    ax.imshow(img)
    ax.imshow(h_img[:, :, 0], cmap=plt.cm.viridis, alpha=0.5)
    ax.set_title(NAMES[idx], fontsize=12, color='black')

def plot_img(ax, img, idx):
    ax.axis('off')
    ax.imshow(img)
    ax.set_title(NAMES[idx], fontsize=12, color='black')

def plot_image(img, pred_hm, valid):
    pred_kp = get_preds(pred_hm, img.shape)
    fig, axs = plt.subplots(3, 8, figsize=(15, 10))
    for j, ax in enumerate(axs.ravel()):
        j-=1
        if j==-1:
            ax.axis('off')
            ax.imshow(img, interpolation='bilinear')
            for k in range(pred_kp.shape[0]):
                ax.scatter(pred_kp[k][0], pred_kp[k][1], s=10, c='red', marker='o')
            ax.set_title("Predictions", fontsize=16, color='black')
        elif valid[j]:
            plot_heatmap(ax, img, pred_hm[:, :, j], j)
        else:
            plot_img(ax, img, j)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def get_preds(hms, input_shape):
    output_shape = hms.shape
    preds = np.zeros((output_shape[-1], 3))
    for j in range(preds.shape[1]):
        hm = hms[:, :, j]
        idx = np.argmax(hm)
        print("idx")
        print(idx)
        print("hm.shape")
        print(hm.shape)
        y, x = np.unravel_index(idx, hm.shape)
        print("unravel_index")
        print(y)
        print(x)
        px = int(math.floor(x + 0.5))
        py = int(math.floor(y + 0.5))
        if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
            diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px]])
            diff = np.sign(diff)
            x += diff[0] * 0.25
            y += diff[1] * 0.25
        preds[j, :2] = [x * input_shape[1] / output_shape[1],
                            y * input_shape[0] / output_shape[0]]
        preds[j, -1] = hm.max() / 255

    return preds
    