import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model
import math
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4

from utils import add_regularization

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def scaling_parameters(input_shape, default_size=224, alpha=1.2, beta=1.1, gamma=1.15):
    size = sum(input_shape[:2]) / 2
    if size <= 240:
        drop_connect_rate = 0.2
    elif size <= 300:
        drop_connect_rate = 0.3
    elif size <= 456:
        drop_connect_rate = 0.4
    else:
        drop_connect_rate = 0.5
    phi = (math.log(size) - math.log(default_size)) / math.log(gamma)
    d = alpha ** phi
    w = beta ** phi
    return d, w, drop_connect_rate

def round_filters(filters, width_coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def EfficientNet(cfg):
    regularizer = l2(cfg.TRAIN.WD)

    if cfg.MODEL.SIZE == 0:
        backbone = EfficientNetB0(include_top=False, input_shape=tuple(cfg.DATASET.INPUT_SHAPE))
    elif cfg.MODEL.SIZE == 1:
        backbone = EfficientNetB1(include_top=False, input_shape=tuple(cfg.DATASET.INPUT_SHAPE))
    elif cfg.MODEL.SIZE == 2:
        backbone = EfficientNetB2(include_top=False, input_shape=tuple(cfg.DATASET.INPUT_SHAPE))
    elif cfg.MODEL.SIZE == 3:
        backbone = EfficientNetB3(include_top=False, input_shape=tuple(cfg.DATASET.INPUT_SHAPE))
    elif cfg.MODEL.SIZE == 4:
        backbone = EfficientNetB4(include_top=False, input_shape=tuple(cfg.DATASET.INPUT_SHAPE))

    backbone = add_regularization(backbone, regularizer)

    d, w, _ = scaling_parameters(cfg.DATASET.INPUT_SHAPE)

    width_coefficient = cfg.MODEL.WIDTH_COEFFICIENT * w
    depth_divisor = cfg.MODEL.DEPTH_DIVISOR
    head_filters = cfg.MODEL.HEAD_CHANNELS
    head_kernel = cfg.MODEL.HEAD_KERNEL
    head_activation = cfg.MODEL.HEAD_ACTIVATION
    keypoints = cfg.DATASET.OUTPUT_SHAPE[-1]
    regularizer = l2(cfg.TRAIN.WD)

    x = backbone.layers[-1].output
    for i in range(cfg.MODEL.HEAD_BLOCKS):
        x = layers.Conv2DTranspose(
            round_filters(head_filters, width_coefficient, depth_divisor),
            head_kernel,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=regularizer,
            name='head_block{}_conv'.format(i + 1))(x)
        x = layers.BatchNormalization(name='head_block{}_bn'.format(i + 1))(x)
        x = layers.Activation(head_activation, name='head_block{}_activation'.format(i + 1))(x)

    x = layers.Conv2D(
        keypoints,
        cfg.MODEL.FINAL_KERNEL,
        padding='same',
        use_bias=True,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        kernel_regularizer=regularizer,
        name='final_conv')(x)

    return Model(backbone.input, x, name=f'EfficientNetLite_{cfg.MODEL.SIZE}')