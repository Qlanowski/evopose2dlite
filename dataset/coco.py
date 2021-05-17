from yacs.config import CfgNode as CN

cn = CN(new_allowed=True)

cn.DATASET = CN(new_allowed=True)
cn.DATASET.NAME = 'COCO'
cn.DATASET.TFRECORDS = r'C:\Users\ulano\source\repos\evopose2dlite\data\tfrecords_foot'
cn.DATASET.ANNOT = r'C:\Users\ulano\source\repos\evopose2dlite\data\annotations\person_keypoints_val2017.json'
cn.DATASET.RUN_EXAMPLES = r'run_examples'
cn.DATASET.TRAIN_SAMPLES = 149813
cn.DATASET.VAL_SAMPLES = 11004
cn.DATASET.KP_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19]
cn.DATASET.KP_UPPER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cn.DATASET.KP_LOWER = []
cn.DATASET.BGR = False
cn.DATASET.NORM = True
cn.DATASET.MEANS = [0.485, 0.456, 0.406]  # imagenet means RGB
cn.DATASET.STDS = [0.229, 0.224, 0.225]
cn.DATASET.INPUT_SHAPE = [256, 192, 3]
cn.DATASET.OUTPUT_SHAPE = [64, 48, 23]
cn.DATASET.SIGMA = 2 * cn.DATASET.OUTPUT_SHAPE[0] / 64
cn.DATASET.FLIP_PROB = 0.5
cn.DATASET.HALF_BODY_PROB = 0.
cn.DATASET.HALF_BODY_MIN_KP = 8
cn.DATASET.SCALE_FACTOR = 0.3
cn.DATASET.ROT_PROB = 0.6
cn.DATASET.ROT_FACTOR = 40
cn.DATASET.CACHE = False
cn.DATASET.BFLOAT16 = False

cn.TRAIN = CN(new_allowed=True)
cn.TRAIN.BATCH_SIZE = 64
cn.TRAIN.BASE_LR = 0.00025
cn.TRAIN.SCALE_LR = True
cn.TRAIN.LR_SCHEDULE = 'warmup_piecewise'
cn.TRAIN.EPOCHS = 210
cn.TRAIN.DECAY_FACTOR = 0.1
cn.TRAIN.DECAY_EPOCHS = [170, 200]
cn.TRAIN.WARMUP_EPOCHS = 0
cn.TRAIN.WARMUP_FACTOR = 0.1
cn.TRAIN.DISP = True
cn.TRAIN.SEED = 0
cn.TRAIN.WD = 1e-5
cn.TRAIN.SAVE_EPOCHS = 0
cn.TRAIN.SAVE_META = False
cn.TRAIN.VAL = True
cn.TRAIN.TEST = False
cn.TRAIN.WANDB_RUN_ID = None

cn.VAL = CN(new_allowed=True)
cn.VAL.BATCH_SIZE = 64
cn.VAL.FLIP = True
cn.VAL.DROP_REMAINDER = False
cn.VAL.SCORE_THRESH = 0.2
cn.VAL.DET = True
cn.VAL.SAVE_BEST = False

cn.MODEL = CN(new_allowed=True)
cn.MODEL.TYPE = 'evopose'
cn.MODEL.LOAD_WEIGHTS = True
cn.MODEL.PARENT = None
cn.MODEL.GENOTYPE = None
cn.MODEL.WIDTH_COEFFICIENT = 1.
cn.MODEL.DEPTH_COEFFICIENT = 1.
cn.MODEL.DEPTH_DIVISOR = 8
cn.MODEL.ACTIVATION = 'swish'
cn.MODEL.HEAD_BLOCKS = 3
cn.MODEL.HEAD_KERNEL = 3
cn.MODEL.HEAD_CHANNELS = 128
cn.MODEL.HEAD_ACTIVATION = 'swish'
cn.MODEL.FINAL_KERNEL = 3
cn.MODEL.SAVE_DIR = 'models'
cn.MODEL.LITE = False
cn.MODEL.DROPOUT = True
cn.MODEL.QUANTIZATION = False

cn.EVAL = CN(new_allowed=True)
cn.EVAL.WANDB_RUNS = 'qlanowski/rangle/runs'
cn.EVAL.NAMES = [
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
cn.EVAL.SIGMA = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
    0.035, 
    0.035,
    0.035,
    0.035,
    0.035, 
    0.035
]

cn.SEARCH = CN(new_allowed=True)


