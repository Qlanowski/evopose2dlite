import cv2
import numpy as np
import tensorflow as tf
import argparse

import dataset.dataloader as dl
import os

from dataset.coco import cn as cfg


KP_PAIRS = [[5, 6], [6, 12], [12, 11], [11, 5],
            [5, 7], [7, 9], [11, 13], [13, 15],
            [6, 8], [8, 10], [12, 14], [14, 16], 
            [15, 17],[15, 19],[17, 18],
            [16, 20],[16, 21],[20, 21]]

def preprocess(img, DATASET):
    img = tf.cast(img, tf.float32)

    if DATASET.NORM:
        img /= 255.
        img -= [[DATASET.MEANS]]
        img /= [[DATASET.STDS]]

    img = tf.image.resize(img, (DATASET.INPUT_SHAPE[0], DATASET.INPUT_SHAPE[1]))

    if DATASET.BFLOAT16:
        img = tf.cast(img, tf.bfloat16)

    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='evopose2d_XS_f32.yaml')
    args = parser.parse_args()

    tf.random.set_seed(0)

    # load the config .yaml file
    cfg.merge_from_file('configs/' + args.cfg)
    
    model = tf.keras.models.load_model('models/{}.h5'.format(args.cfg.split('.yaml')[0]), compile=False)
    imgs = dl.prediction_examples(model, cfg)
    for img in imgs:
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("image", opencvImage)
        cv2.waitKey()
        cv2.destroyAllWindows()