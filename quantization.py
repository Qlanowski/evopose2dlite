import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import os.path as osp
from utils import detect_hardware
from dataset.dataloader import load_representative_tfds, visualize, prediction_tf_lite, prediction_examples
from dataset.coco import cn as cfg
import argparse
from lr_schedules import WarmupCosineDecay
import cv2
import math

class MyModule(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(shape=(256, 192, 3), dtype=tf.float32)])
  def score(self, image):
    input_shape = [256, 192, 3]
    output_shape = [64, 48, 23]
    hms = self.model(image)
    preds = np.zeros((hms.shape[0], output_shape[-1], 3))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm = hms[i, :, :, j]
            idx = hm.argmax()
            y, x = np.unravel_index(idx, hm.shape)
            px = int(math.floor(x + 0.5))
            py = int(math.floor(y + 0.5))
            if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                diff = np.sign(diff)
                x += diff[0] * 0.25
                y += diff[1] * 0.25
            preds[i, j, :2] = [x * input_shape[1] / output_shape[1],
                              y * input_shape[0] / output_shape[0]]
            preds[i, j, -1] = hm.max() / 255


    return preds

def hm_to_kp(hms, input_shape, output_shape):
    input_shape = [256, 192, 3]
    output_shape = [64,48,23]
    preds = np.zeros((output_shape[-1], 3))
    for j in range(preds.shape[0]):
        hm = hms[:, :, j]
        idx = hm.argmax()
        y, x = np.unravel_index(idx, hm.shape)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True)
    args = parser.parse_args()

    tpu, strategy = detect_hardware(None)
    if tpu:
        cfg.TRAIN.ACCELERATOR = args.tpu
    else:
        cfg.TRAIN.ACCELERATOR = 'GPU/CPU'
    cfg.merge_from_file('configs/' + args.cfg)
    cfg.MODEL.NAME = args.cfg.split('.yaml')[0]

    if cfg.DATASET.OUTPUT_SHAPE[-1] == 17:
        cfg.DATASET.KP_FLIP = cfg.DATASET.KP_FLIP[:17]

    cfg.MODEL.TFRECORDS = 'data/tfrecords_foot/val'
    train_ds = load_representative_tfds(cfg)

    def representative_dataset():
        for img in train_ds:
            yield [img]

    def representative_dataset_gen():
        num_calibration_images = 200
        for i in range(num_calibration_images):
            image = tf.random.normal([1] + cfg.DATASET.INPUT_SHAPE)
            yield [image]

    ds = iter(train_ds)
    l = []
    for i, img in enumerate(ds):
        l.append(img)

    def representative_dataset2():
        for img in l:
            yield [img]
    

    model = tf.keras.models.load_model(r"C:\Users\ulano\source\repos\evopose2dlite\models\eflite_0_f32.h5", 
            custom_objects={
                        'relu6': tf.nn.relu6,
                        'WarmupCosineDecay': WarmupCosineDecay
            })

    

    # images = prediction_examples(model, cfg)
    # for img in images:
    #     opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #     cv2.imshow('', opencvImage)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset2
    tflite_quant_model = converter.convert()

    TFLITE_FILE_PATH = 'models/eflite0_int.tflite'

    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_quant_model)

    images = prediction_tf_lite(TFLITE_FILE_PATH, cfg)
    for img in images:
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow('', opencvImage)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
    # interpreter.allocate_tensors()

    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # input_shape = input_details[0]['shape']
    # for img in train_ds:
    #     input_data = np.array([img.numpy()], dtype=np.float32)
    #     interpreter.set_tensor(input_details[0]['index'], input_data)
    #     interpreter.invoke()
    #     output_data = interpreter.get_tensor(output_details[0]['index'])
    #     kp = hm_to_kp(output_data[0],None,None)
    #     img = img[:,:,::-1]
    #     cv2.imshow('', visualize(np.uint8(img), kp[:, :2], kp[:, -1]))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    # ds = load_tfds(cfg, 'val', det=False, predict_kp=True, drop_remainder=False, visualize=True)
    # for i, (ids, imgs, kps, Ms, scores, hms, valids) in enumerate(ds):
    #     f = 18 * 3 - 1
    #     for i in range(cfg.TRAIN.BATCH_SIZE):
    #         kp = kps[i]
    #         if np.sum(kp[:,2][17:]) > 0:
    #             img = imgs[i]
    #             pl.plot_image(np.uint8(img), hms[i], kp[:, -1].numpy())
    #             cv2.imshow('', dl.visualize(np.uint8(img), kp[:, :2].numpy(), kp[:, -1].numpy()))
    #             cv2.waitKey()
    #             cv2.destroyAllWindows()

    

    # interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)

    # my_signature = interpreter.get_signature_runner()

    # for img in train_ds:
    #     output = my_signature(x=tf.constant([img], dtype=tf.float32))
    #     print('ok')
    


    print("as")