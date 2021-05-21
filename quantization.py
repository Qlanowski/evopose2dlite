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
import math

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


    def representative_dataset2():
        for img in l:
            yield [img]

    def get_preds(hms):
        hms = tf.cast(hms, dtype=tf.float32)
        output_shape = hms.shape[1:]
        preds = np.zeros((output_shape[-1], 3))
        for j in range(preds.shape[0]):
            hm = hms[:, :, j]
            f_hm = tf.nest.flatten(hm)
            idx = tf.math.argmax(f_hm)
            y, x = tf.unravel_index(idx, output_shape[0:2])
            px = int(math.floor(x + 0.5))
            py = int(math.floor(y + 0.5))
            if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                    hm[py + 1][px] - hm[py - 1][px]])
                diff = np.sign(diff)
                x += diff[0] * 0.25
                y += diff[1] * 0.25
            preds[j, :2] = [x, y]
            preds[j, -1] = np.array(hm).max() / 255

        return preds
    
    @tf.function
    def preprocess(img):
        img = tf.cast(img, dtype=tf.float32)
        DATASET_MEANS = [0.485, 0.456, 0.406]  # imagenet means RGB
        DATASET_STDS = [0.229, 0.224, 0.225]

        img /= 255.
        img -= [[DATASET_MEANS]]
        img /= [[DATASET_STDS]]
        return img
    print(tf.__version__)
    model = tf.keras.models.load_model(r"C:\Users\ulano\source\repos\evopose2dlite\models\eflite4_bf16_hb.h5", 
            custom_objects={
                        'relu6': tf.nn.relu6,
                        'WarmupCosineDecay': WarmupCosineDecay
            })
    # # new_model = tf.keras.Sequential()
    # # new_model.add(tf.keras.layers.Lambda(preprocess))
    # # new_model.add(tf.keras.layers.experimental.preprocessing.Resizing(cfg.DATASET.INPUT_SHAPE[0], cfg.DATASET.INPUT_SHAPE[1]))
    # # new_model.add(model)
    # # new_model.add(tf.keras.layers.Lambda(get_preds))

    # images = prediction_examples(model, cfg)
    # for img in images:
    #     opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #     cv2.imshow('', opencvImage)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset2
    tflite_quant_model = converter.convert()

    TFLITE_FILE_PATH = 'models/eflite0_float16_23.tflite'

    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_quant_model)

    # images = prediction_tf_lite(TFLITE_FILE_PATH, cfg)
    # for img in images:
    #     opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #     cv2.imshow('', opencvImage)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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