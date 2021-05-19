import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import cv2
import shutil
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import os.path as osp
from utils import get_flops, detect_hardware
from dataset.dataloader import load_tfds, prediction_examples
from dataset.coco import cn as cfg
from nets.simple_basline import SimpleBaseline
from nets.hrnet import HRNet
from nets.evopose2d import EvoPose
from nets.efficient_net_lite import EfficientNetLite
from nets.efficient_net import EfficientNet
from time import time
import argparse
from validate import validate

import wandb
from wandb.keras import WandbCallback


@tf.function
def mse(y_actual, y_pred):
    y_actual = tf.cast(y_actual, dtype=y_pred.dtype)
    valid = tf.cast(tf.math.reduce_max(y_actual, axis=(1, 2)) > 0, dtype=y_pred.dtype)
    valid_mask = tf.reshape(valid, [tf.shape(y_actual)[0], 1, 1, tf.shape(valid)[-1]])
    return tf.reduce_mean(tf.square(y_actual - y_pred) * valid_mask)


def setup_wandb(cfg, model):
    key_path = "wandb_api_key.txt"
    if os.path.isfile(key_path):
        with open (key_path, "r") as key_file:
            key = key_file.readlines()[0]
            os.environ["WANDB_API_KEY"] = key

    parameters = '{:.2f}M'.format(model.count_params()/1e6)
    flops = '{:.2f}G'.format(get_flops(model)/2/1e9)

    group = ''
    if cfg.DATASET.OUTPUT_SHAPE[-1] == 23 and cfg.TRAIN.TEST:
        group = "test body+foot"
    elif cfg.DATASET.OUTPUT_SHAPE[-1] == 23 and cfg.TRAIN.TEST == False:
        group = "valid body+foot"
    elif cfg.DATASET.OUTPUT_SHAPE[-1] == 17 and cfg.TRAIN.TEST:
        group = "test body"
    elif cfg.DATASET.OUTPUT_SHAPE[-1] == 17 and cfg.TRAIN.TEST == False:
        group = "valid body"

    wandb.init(
        project="rangle",
        group = group,
        id = cfg.TRAIN.WANDB_RUN_ID if cfg.TRAIN.WANDB_RUN_ID else None,
        resume = True if cfg.TRAIN.WANDB_RUN_ID else False,
        config = {
            "model": cfg.MODEL.NAME,
            "flops": flops,
            "parameters": parameters,
            "lr_schedule": cfg.TRAIN.LR_SCHEDULE,
            "batch_size": cfg.TRAIN.BATCH_SIZE,
            "epoch": cfg.TRAIN.EPOCHS,
            "input_shape": '_'.join(str(x) for x in cfg.DATASET.INPUT_SHAPE),
            "output_shape": '_'.join(str(x) for x in cfg.DATASET.OUTPUT_SHAPE),
            "number_of_joints": cfg.DATASET.OUTPUT_SHAPE[-1],
            "loss": "mean_from_visible",
            "optimizer": "Adam",
            "train_samples": cfg.DATASET.TRAIN_SAMPLES,
            "val_samples": cfg.DATASET.VAL_SAMPLES,
            "transfer": cfg.MODEL.LOAD_WEIGHTS,
            "bfloat16": cfg.DATASET.BFLOAT16,
            "half body prob": cfg.DATASET.HALF_BODY_PROB
        })
    shutil.copy2(f'configs/{cfg.MODEL.NAME}.yaml', os.path.join(wandb.run.dir, "model_config.yaml"))

    return wandb.config

def train(strategy, cfg):
    os.makedirs(cfg.MODEL.SAVE_DIR, exist_ok=True)

    if cfg.DATASET.BFLOAT16:
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_policy(policy)

    tf.random.set_seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)

    spe = int(np.ceil(cfg.DATASET.TRAIN_SAMPLES / cfg.TRAIN.BATCH_SIZE))
    spv = cfg.DATASET.VAL_SAMPLES // cfg.VAL.BATCH_SIZE

    if cfg.TRAIN.SCALE_LR:
        lr = cfg.TRAIN.BASE_LR * cfg.TRAIN.BATCH_SIZE / 32
        cfg.TRAIN.WARMUP_FACTOR = 32 / cfg.TRAIN.BATCH_SIZE
    else:
        lr = cfg.TRAIN.BASE_LR

    if cfg.TRAIN.LR_SCHEDULE == 'warmup_cosine_decay':
        lr_schedule = WarmupCosineDecay(
            initial_learning_rate=lr,
            decay_steps=cfg.TRAIN.EPOCHS * spe,
            warmup_steps=cfg.TRAIN.WARMUP_EPOCHS * spe,
            warmup_factor=cfg.TRAIN.WARMUP_FACTOR)
    elif cfg.TRAIN.LR_SCHEDULE == 'warmup_piecewise':
        lr_schedule = WarmupPiecewise(
            boundaries=[x * spe for x in cfg.TRAIN.DECAY_EPOCHS],
            values=[lr, lr / 10, lr / 10 ** 2],
            warmup_steps=spe * cfg.TRAIN.WARMUP_EPOCHS,
            warmup_factor=cfg.TRAIN.WARMUP_FACTOR)
    else:
        lr_schedule = lr

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        if cfg.TRAIN.WANDB_RUN_ID:
            api = wandb.Api()
            run = api.run(f"{cfg.EVAL.WANDB_RUNS}/{cfg.TRAIN.WANDB_RUN_ID}")
            run.file("model-best.h5").download(replace=True)
            model = tf.keras.models.load_model('model-best.h5', 
                custom_objects={
                            'relu6': tf.nn.relu6,
                            'WarmupCosineDecay': WarmupCosineDecay
                })
            model.compile(optimizer=model.optimizer, loss=mse)
        else:
            if cfg.MODEL.TYPE == 'simple_baseline':
                model = SimpleBaseline(cfg)
            elif cfg.MODEL.TYPE == 'hrnet':
                model = HRNet(cfg)
            elif cfg.MODEL.TYPE == 'evopose':
                model = EvoPose(cfg)
            elif cfg.MODEL.TYPE == 'eflite':
                model = EfficientNetLite(cfg)
            elif cfg.MODEL.TYPE == 'ef':
                model = EfficientNet(cfg)

            model.compile(optimizer=optimizer, loss=mse)

    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]
    cfg.DATASET.SIGMA = 2 * cfg.DATASET.OUTPUT_SHAPE[0] / 64

    wandb_config = setup_wandb(cfg, model)

    train_ds = load_tfds(cfg, 'train')
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    if cfg.TRAIN.VAL:
        val_ds = load_tfds(cfg, 'val')
        val_ds = strategy.experimental_distribute_dataset(val_ds)

    print('Training {} ({} / {}) on {} for {} epochs'
          .format(cfg.MODEL.NAME, wandb_config.parameters,
                  wandb_config.flops, cfg.TRAIN.ACCELERATOR, cfg.TRAIN.EPOCHS))

    initial_epoch = 0
    if cfg.TRAIN.WANDB_RUN_ID:
        initial_epoch = cfg.TRAIN.INITIAL_EPOCH

    model.fit(train_ds, initial_epoch=initial_epoch, epochs=cfg.TRAIN.EPOCHS, verbose=1,
                        validation_data=val_ds, 
                        validation_steps=spv, 
                        steps_per_epoch=spe,
                        callbacks=[WandbCallback()])
    
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--tpu', default=None)
    parser.add_argument('--val', type=int, default=1)
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    tpu, strategy = detect_hardware(args.tpu)
    if tpu:
        cfg.TRAIN.ACCELERATOR = args.tpu
    else:
        cfg.TRAIN.ACCELERATOR = 'GPU/CPU'
    cfg.merge_from_file('configs/' + args.cfg)
    cfg.MODEL.NAME = args.cfg.split('.yaml')[0]

    if cfg.DATASET.OUTPUT_SHAPE[-1] == 17:
        cfg.DATASET.KP_FLIP = cfg.DATASET.KP_FLIP[:17]

    cfg.TRAIN.TEST = args.test

    model = train(strategy, cfg)

    model = tf.keras.models.load_model(
        os.path.join(wandb.run.dir, "model-best.h5"), 
        custom_objects={
            'relu6': tf.nn.relu6,
            'WarmupCosineDecay': WarmupCosineDecay
        })

    cfg.VAL.DROP_REMAINDER = False
    if args.val == 1:
        if cfg.DATASET.OUTPUT_SHAPE[-1] == 23:
            mAP, AP_50, AP_75, AP_small, AP_medium, AP_large = validate(strategy, cfg, model, clear_foot=False)
            print('AP body+foot: {:.5f}'.format(mAP))
            wandb.log({
                'mAP_body+foot': mAP,
                'AP_50_body+foot': AP_50,
                'AP_75_body+foot': AP_75,
                'AP_small_body+foot': AP_small,
                'AP_medium_body+foot': AP_medium,
                'AP_large_body+foot': AP_large
            })

        mAP, AP_50, AP_75, AP_small, AP_medium, AP_large = validate(strategy, cfg, model, clear_foot=True)
        print('AP body: {:.5f}'.format(mAP))
        wandb.log({
            'mAP_body': mAP,
            'AP_50_body': AP_50,
            'AP_75_body': AP_75,
            'AP_small_body': AP_small,
            'AP_medium_body': AP_medium,
            'AP_large_body': AP_large
        })

    imgs = prediction_examples(model, cfg)
    images = [wandb.Image(img) for img in imgs]
    wandb.log({"runners": images})
