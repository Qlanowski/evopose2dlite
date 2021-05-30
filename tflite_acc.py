import cv2
import numpy as np
import tensorflow as tf
import argparse
import dataset.plots as pl
import dataset.dataloader as dl
from validate import validate_tflite
from lr_schedules import WarmupCosineDecay
from dataset.dataloader import load_tfds

tf.random.set_seed(0)

from dataset.coco import cn as cfg
# cfg.DATASET.INPUT_SHAPE = [512, 384, 3]
cfg.DATASET.NORM = False
cfg.DATASET.BGR = True
cfg.DATASET.HALF_BODY_PROB = 0
cfg.MODEL.TFRECORDS = 'gs://rangle/tfrecords_foot'
# cfg.DATASET.VAL_SAMPLES = 10
cfg.VAL.BATCH_SIZE = 1
cfg.TRAIN.BATCH_SIZE = 1
cfg.DATASET.ANNOT = './data/annotations'

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default=None)
parser.add_argument('--model', default=None)
parser.add_argument('--results', default=None)
args = parser.parse_args()


def quant_int(folder, model_name, input_size):
    cfg.DATASET.INPUT_SHAPE = input_size
    ds = dl.load_tfds(cfg, 'val', det=False, predict_kp=True, drop_remainder=False, visualize=True)
    def represent():
        for i, (ids, imgs, kps, Ms, scores, hms, valids) in enumerate(ds):
            if i < 100:
                img = imgs.numpy()
                yield [img]

    model_path = f'{folder}/{model_name}.h5'
    model = tf.keras.models.load_model(model_path, 
            custom_objects={
                        'relu6': tf.nn.relu6,
                        'WarmupCosineDecay': WarmupCosineDecay
            })
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = represent
    tflite_quant_model = converter.convert()

    TFLITE_FILE_PATH = f'{folder}/{model_name}_int.tflite'

    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_quant_model)

# quant_int('models/eflite', 'eflite_0_f32_1s438o9s', [224,224,3])
# quant_int('models/eflite', 'eflite_4_f32_2aqxr0tj', [224,224,3])

# quant_int('models/evo', 'evo_XS_f32_3k83fz2x', [256, 192, 3])
# quant_int('models/evo', 'evo_S_f32_2z5u2n9h', [256, 192, 3])
# quant_int('models/evo', 'evo_M_f32_2mknlo85', [384, 288, 3])

# quant_int('models/evolite', 'evolite_XS_f32_3bxk1z2i', [256, 192, 3])
# quant_int('models/evolite', 'evolite_S_f32_2q6wypd1', [256, 192, 3])
# quant_int('models/evolite', 'evolite_M_f32_24yaz1ej', [384, 288, 3])

def calculate_acc(cfg, path, result_path):
    interpreter = tf.lite.Interpreter(path)
    a = validate_tflite(cfg, result_path, interpreter, split='val', clear_foot=False)
    print(a)


cfg.MODEL.NAME = args.model
calculate_acc(cfg, f'{args.folder}/{args.model}.tflite', f'{args.results}/{args.model}.json')

# ds = load_tfds(cfg, 'val', det=False,
#                    predict_kp=True, drop_remainder=cfg.VAL.DROP_REMAINDER)

# batches = set([])
# for count, batch in enumerate(ds):
#     ids, imgs, _, Ms, scores, kp_id = batch
#     id = ids.numpy()[0]
#     batches.add(str(kp_id.numpy()[0]))

# ds = load_tfds(cfg, 'val', det=False,
#                    predict_kp=True, drop_remainder=cfg.VAL.DROP_REMAINDER)

# batches2 = set([])
# for count, batch in enumerate(ds):
#     ids, imgs, _, Ms, scores, kp_id = batch
#     id = ids.numpy()[0]
#     code = str(kp_id.numpy()[0])
#     batches2.add(code)
#     if code not in batches:
#         print('ups')

# print("success")



