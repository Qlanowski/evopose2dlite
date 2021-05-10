import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import os.path as osp
from utils import get_flops, detect_hardware
from dataset.dataloader import load_tfds
from dataset.coco import cn as cfg
from nets.simple_basline import SimpleBaseline
from nets.hrnet import HRNet
from nets.evopose2d import EvoPose
from time import time
import pickle
import argparse
from validate import validate

import wandb
from wandb.keras import WandbCallback


@tf.function
def mse(y_actual, y_pred):
    valid = tf.cast(tf.math.reduce_max(y_actual, axis=(1, 2)) > 0, dtype=tf.float32)
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

    wandb.init(
        project="rangle",
        group=f'{cfg.MODEL.NAME}_{cfg.DATASET.OUTPUT_SHAPE[-1]}',
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
            "transfer": cfg.MODEL.LOAD_WEIGHTS
        })
    return wandb.config

def train(strategy, cfg):
    os.makedirs(cfg.MODEL.SAVE_DIR, exist_ok=True)

    if cfg.DATASET.BFLOAT16:
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_policy(policy)

    tf.random.set_seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)

    meta_data = {'train_loss': [], 'val_loss': [], 'config': cfg}

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
        if cfg.MODEL.TYPE == 'simple_baseline':
            model = SimpleBaseline(cfg)
        elif cfg.MODEL.TYPE == 'hrnet':
            model = HRNet(cfg)
        elif cfg.MODEL.TYPE == 'evopose':
            model = EvoPose(cfg)

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

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./models/{cfg.MODEL.NAME}/model.h5',
        save_weights_only=True,
        monitor=['val_loss'],
        mode='min',
        save_best_only=True)

    model.fit(train_ds, epochs=cfg.TRAIN.EPOCHS, verbose=1,
                        validation_data=val_ds, 
                        validation_steps=spv, 
                        steps_per_epoch=spe,
                        callbacks=[WandbCallback()])

    # os.mkdir(f'./models/{cfg.MODEL.NAME}')
    # with open(f'./models/{cfg.MODEL.NAME}/training.history', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--tpu', default=None)
    parser.add_argument('--val', default=1)
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

    model = train(strategy, cfg)

    if args.val:
        mAP, AP_50, AP_75, AP_small, AP_medium, AP_large = validate(strategy, cfg, model)
        print('AP: {:.5f}'.format(AP))
        wandb.log({
            'mAP_org': mAP,
            'AP_50_org': AP_50,
            'AP_75_org': AP_75,
            'AP_small_org': AP_small,
            'AP_medium_org': AP_medium,
            'AP_large_org': AP_large
        })
