import cv2
import numpy as np
import tensorflow as tf
import argparse

import dataset.plots as pl
import dataset.dataloader as dl
import wandb
import os
from PIL import Image

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

def start_wandb():
    key_path = "wandb_api_key.txt"
    if os.path.isfile(key_path):
        with open (key_path, "r") as key_file:
            key = key_file.readlines()[0]
            os.environ["WANDB_API_KEY"] = key

    wandb.init(project="images", group = "testing")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='evopose2d_XS_f32.yaml')
    args = parser.parse_args()

    tf.random.set_seed(0)

    # load the config .yaml file
    cfg.merge_from_file('configs/' + args.cfg)
    
    # start_wandb()

    model = tf.keras.models.load_model('models/{}.h5'.format(args.cfg.split('.yaml')[0]))

    imgs = []
    for img_path in os.listdir('run_examples'):
        img_bytes = open(os.path.join('run_examples', img_path), 'rb').read()
        img_org = tf.image.decode_jpeg(img_bytes, channels=3)
        img = preprocess(img_org, cfg.DATASET)
        hms = model.predict(tf.expand_dims(img, 0))
        hms = tf.squeeze(hms)
        preds = pl.get_preds(hms, img_org.shape)

        img_cv = cv2.imread(os.path.join('run_examples', img_path))
        overlay = img_cv.copy()

        for i, (x, y, v) in enumerate(preds):
            overlay = cv2.circle(overlay, (int(np.round(x)), int(np.round(y))), 2, [255, 255, 255], 2)

        for p in KP_PAIRS:
            if len(preds) > p[0] and len(preds) > p[1]:
                overlay = cv2.line(overlay,
                                tuple(np.int32(np.round(preds[p[0], :2]))),
                                tuple(np.int32(np.round(preds[p[1], :2]))), [255, 255, 255], 2)

        img = cv2.addWeighted(overlay, 0.8, img_cv, 1 - 0.8, 0)
        im_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_pil)
        imgs.append(im_pil)

        cv2.imshow(img_path, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # images = [wandb.Image(img) for img in imgs]
    # wandb.log({"example": images})