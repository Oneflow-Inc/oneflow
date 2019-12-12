from yacs.config import CfgNode as CN
_C = CN()


_C.TRAINING = True
# ---------------------------------------------------------------------------- #
#  Env
# ---------------------------------------------------------------------------- #
_C.ENV = CN()
_C.ENV.NUM_GPUS = 1
_C.ENV.ENABLE_INPLACE = False
_C.ENV.CUDNN_BUFFER_SIZE_LIMIT = 1280
_C.ENV.CUDNN_CONV_HEURISTIC_SEARCH_ALGO = True
_C.ENV.CUDNN_CONV_USE_DETERMINISTIC_ALGO_ONLY = False

# ---------------------------------------------------------------------------- #
#  Decoder
# ---------------------------------------------------------------------------- #
_C.DECODER = CN()
# Anchor generator params
_C.DECODER.FPN_LAYERS = 5
_C.DECODER.FEATURE_MAP_STRIDE = 4
_C.DECODER.ASPECT_RATIOS = [0.5, 1.0, 2.0]
_C.DECODER.ANCHOR_SCALES = 32

# ---------------------------------------------------------------------------- #
#  Backbone
# ---------------------------------------------------------------------------- #
_C.BACKBONE = CN()
_C.BACKBONE.CONV_BODY = "R-50-FPN"
_C.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.BACKBONE.RESNET_STEM_OUT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
#  RPN
# ---------------------------------------------------------------------------- #
_C.RPN = CN()
_C.RPN.RPN_HEAD = "SingleConvRPNHead"

# Anchor targets params
_C.RPN.POSITIVE_OVERLAP_THRESHOLD = 0.7
_C.RPN.NEGATIVE_OVERLAP_THRESHOLD = 0.3
_C.RPN.SUBSAMPLE_NUM_PER_IMG = 256
_C.RPN.FOREGROUND_FRACTION = 0.5
_C.RPN.WEIGHT_X = 1.0
_C.RPN.WEIGHT_Y = 1.0
_C.RPN.WEIGHT_W = 1.0
_C.RPN.WEIGHT_H = 1.0
_C.RPN.RANDOM_SUBSAMPLE = False

# RPN post processor
_C.RPN.TOP_N_PER_FM_TRAIN = 2000
_C.RPN.NMS_TOP_N_TRAIN = 2000
_C.RPN.TOP_N_PER_IMG_TRAIN = 1000
_C.RPN.TOP_N_PER_FM_TEST = 2000
_C.RPN.NMS_TOP_N_TEST = 2000
_C.RPN.TOP_N_PER_IMG_TEST = 2000
_C.RPN.NMS_THRESH = 0.7

# ---------------------------------------------------------------------------- #
#  BOX Head
# ---------------------------------------------------------------------------- #
_C.BOX_HEAD = CN()
_C.BOX_HEAD.FOREGROUND_THRESHOLD = 0.5
_C.BOX_HEAD.BACKGROUND_THRESHOLD_LOW = 0.0
_C.BOX_HEAD.BACKGROUND_THRESHOLD_HIGH = 0.5
_C.BOX_HEAD.FOREGROUND_FRACTION = 0.25
_C.BOX_HEAD.WEIGHT_X = 10.0
_C.BOX_HEAD.WEIGHT_Y = 10.0
_C.BOX_HEAD.WEIGHT_W = 5.0
_C.BOX_HEAD.WEIGHT_H = 5.0
_C.BOX_HEAD.NUM_CLASSES = 81
_C.BOX_HEAD.NUM_SAMPLED_ROI_PER_IMG = 512
_C.BOX_HEAD.POOLED_H = 7
_C.BOX_HEAD.POOLED_W = 7
_C.BOX_HEAD.SPATIAL_SCALE = 0.25
_C.BOX_HEAD.SAMPLING_RATIO = 2
_C.BOX_HEAD.RANDOM_SUBSAMPLE = False

# ---------------------------------------------------------------------------- #
#  Mask Head
# ---------------------------------------------------------------------------- #
_C.MASK_HEAD = CN()
_C.MASK_HEAD.POOLED_H = 14
_C.MASK_HEAD.POOLED_W = 14
_C.MASK_HEAD.SAMPLING_RATIO = 2
_C.MASK_HEAD.SPATIAL_SCALE = 0.25

# ---------------------------------------------------------------------------- #
#  Train & Eval
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.IMAGE_PER_GPU = 2
_C.TRAIN.MODEL_INIT_PATH = "/model_zoo/detection/R-50"

_C.TRAIN.DATASET = CN()
_C.TRAIN.DATASET.DATASET_DIR = "/dataset/mscoco_2017"
_C.TRAIN.DATASET.ANNOTATION = "annotations/instances_train2017.json"
_C.TRAIN.DATASET.IMAGE_DIR = "train2017"
_C.TRAIN.DATASET.SHUFFLE = True
_C.TRAIN.DATASET.RANDOM_SEED = 123456
_C.TRAIN.DATASET.MAX_SEGM_POLY_POINTS_PER_IMAGE = 65536

_C.TRAIN.INPUT = CN()
_C.TRAIN.INPUT.TARGET_SIZE = 800
_C.TRAIN.INPUT.MAX_SIZE = 1333
_C.TRAIN.INPUT.IMAGE_ALIGN_SIZE = 32
_C.TRAIN.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
_C.TRAIN.INPUT.PIXEL_STD = [1., 1., 1.]
_C.TRAIN.INPUT.MIRROR_PROB = 0.5
_C.TRAIN.INPUT.MAX_BOXES_PER_IMAGE = 128
_C.TRAIN.INPUT.MAX_POLYGONS_PER_OBJECT = 2
_C.TRAIN.INPUT.MAX_POINTS_PER_POLYGON = 256

_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 1
_C.EVAL.IMAGE_PER_GPU = 1
_C.EVAL.MODEL_LOAD_PATH = "/model_zoo/detection/mask_rcnn_R_50_FPN_1x_with_momentum"

_C.EVAL.DATASET = CN()
_C.EVAL.DATASET.DATASET_DIR = "/dataset/mscoco_2017"
_C.EVAL.DATASET.ANNOTATION = "annotations/instances_val2017.json"
_C.EVAL.DATASET.IMAGE_DIR = "val2017"

_C.EVAL.INPUT = CN()
_C.EVAL.INPUT.TARGET_SIZE = 400
_C.EVAL.INPUT.MAX_SIZE = 600
_C.EVAL.INPUT.IMAGE_ALIGN_SIZE = 32
_C.EVAL.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
_C.EVAL.INPUT.PIXEL_STD = [1., 1., 1.]

_C.SOLVER = CN()
_C.SOLVER.PRIMARY_LR = 0.00125
_C.SOLVER.SECONDARY_LR = 0.0025
_C.SOLVER.OPTIMIZER = "momentum"  # "momentum" or "sgd"
_C.SOLVER.MOMENTUM_BETA = 0.9
_C.SOLVER.WEIGHT_L2 = 0.0001
_C.SOLVER.BIAS_L2 = 0.0
_C.SOLVER.ENABLE_LR_DECAY = True
_C.SOLVER.LR_DECAY_BOUNDARIES = [960000, 1280000]
_C.SOLVER.LR_DECAY_VALUES = [0.01, 0.001, 0.0001]
_C.SOLVER.ENABLE_WARMUP = False
_C.SOLVER.WARMUP_METHOD = "linear"  # "linear" or "constant"
_C.SOLVER.WARMUP_BATCHES = 500
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3

# ---------------------------------------------------------------------------- #
#  Debug
# ---------------------------------------------------------------------------- #
_C.DEBUG = CN()
_C.DEBUG.RPN_RANDOM_SAMPLE = True
_C.DEBUG.ROI_HEAD_RANDOM_SAMPLE = True


def get_default_cfgs():
    """
    Get a yacs CfgNode object with default values for maskrcnn object
    """
    return _C.clone()


def update_by_path(node, path, value):
    last_idx = len(path) - 1
    for i, key in enumerate(path):
        if i == last_idx:
            # print("updating", id(node), key)
            node[key] = value
        else:
            if key not in node:
                node[key] = CN()
            node = node[key]


def compare_config(d1, d2, path=None, d1_diff=None, d2_diff=None):
    if d1_diff is None:
        assert(path is None)
        d1_diff = CN()
    if d2_diff is None:
        assert(path is None)
        d2_diff = CN()
    if path is None:
        path = []
    for k in d1.keys():
        if k not in d2:
            raise ValueError

        if type(d1[k]) is CN:
            sub_path = None
            if path == []:
                sub_path = [k]
            else:
                sub_path = path + [k]
            assert sub_path is not None
            compare_config(d1[k], d2[k], sub_path, d1_diff, d2_diff)
        else:
            if d1[k] != d2[k]:
                print("{}:".format(".".join(path)))
                print("- {} : {}".format(k, d1[k]))
                print("+ {} : {}".format(k, d2[k]))
                update_by_path(d1_diff, path + [k], d1[k])
                update_by_path(d2_diff, path + [k], d2[k])
    return (d1_diff, d2_diff)
