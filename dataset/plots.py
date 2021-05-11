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

def get_preds(hms, img_shape):
    hms = hms.numpy()
    preds = np.zeros((hms.shape[-1], 2))
    sh = img_shape[0] / hms.shape[0]
    sw = img_shape[1] / hms.shape[1]

    for j in range(hms.shape[-1]):
        hm = hms[:, :, j]
        idx = hm.argmax()
        max_y, max_x = np.unravel_index(idx, hm.shape)
        hms[max_y, max_x, j] = float('-inf')

        idx = hm.argmax()
        sec_y, sec_x = np.unravel_index(idx, hm.shape)

        diff = math.sqrt(((max_y-sec_y)**2)+((max_x-sec_x)**2))

        dy = (sec_y - max_y)/diff
        dx = (sec_x - max_x)/diff

        x = max_x + 0.25 * dx
        y = max_y + 0.25 * dy
        preds[j, 0] = x * sw
        preds[j, 1] = y * sh

    return preds
    