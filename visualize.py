import cv2
import numpy as np
import tensorflow as tf

import dataset.plots as pl
import dataset.dataloader as dl

tf.random.set_seed(0)

from dataset.coco import cn as cfg
cfg.DATASET.INPUT_SHAPE = [512, 384, 3]
cfg.DATASET.NORM = False
cfg.DATASET.BGR = True
cfg.DATASET.HALF_BODY_PROB = 1.

ds = dl.load_tfds(cfg, 'val', det=False, predict_kp=True, drop_remainder=False, visualize=True)
for i, (ids, imgs, kps, Ms, scores, hms, valids) in enumerate(ds):
    f = 18 * 3 - 1
    for i in range(cfg.TRAIN.BATCH_SIZE):
        kp = kps[i]
        if np.sum(kp[:,2][17:]) > 0:
            img = imgs[i]
            pl.plot_image(np.uint8(img), hms[i], kp[:, -1].numpy())
            cv2.imshow('', dl.visualize(np.uint8(img), kp[:, :2].numpy(), kp[:, -1].numpy()))
            cv2.waitKey()
            cv2.destroyAllWindows()
