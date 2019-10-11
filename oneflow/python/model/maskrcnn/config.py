from yacs.config import CfgNode as CN
_C = CN()

_C.TRAINING = True
# ---------------------------------------------------------------------------- #
#  Decoder
# ---------------------------------------------------------------------------- #
_C.DECODER = CN()
# Dataset for train and test
_C.DECODER.DATA_DIR_TRAIN = "/dataset/mask_rcnn/sample_1_train"
_C.DECODER.DATA_DIR_TEST = "/dataset/mask_rcnn/sample_1_train"
# Static shape to store images, should walk through the dataset to find it
_C.DECODER.IMAGE_STATIC_SIZE_TRAIN = [1344, 800]
_C.DECODER.IMAGE_STATIC_SIZE_TEST = [1344, 800]
# Preprocess params: target resize
_C.DECODER.RESIZE_TARGET_SIZE_TRAIN = 800
_C.DECODER.RESIZE_MAX_SIZE_TRAIN = 1333
_C.DECODER.RESIZE_TARGET_SIZE_TEST = 800
_C.DECODER.RESIZE_MAX_SIZE_TEST = 1333
# Preprocess params: norm by channel
_C.DECODER.PREPROCESS_NORMAL_MEAN = [102.9801, 115.9465, 122.7717]
_C.DECODER.PREPROCESS_NORMAL_STD = [1.0, 1.0, 1.0]
# Max number of ground truth boxes one image should have
_C.DECODER.MAX_NUM_OF_GT_BBOXES_PER_IMAGE = 256
# Max bytesize to store segmentations for each image
_C.DECODER.MAX_SEGM_BYTES_SIZE_PER_IMAGE = 1048576
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
_C.RPN.TOP_N_PER_IMG_TRAIN = 2000
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
#  Training Configs
# ---------------------------------------------------------------------------- #
_C.TRAINING_CONF = CN()
_C.TRAINING_CONF.IMG_PER_GPU = 2
_C.TRAINING_CONF.PIECE_SIZE = 2
_C.TRAINING_CONF.BATCH_SIZE = 2
_C.TRAINING_CONF.TOTAL_BATCH_NUM = 1440000
_C.TRAINING_CONF.PRIMARY_LR = 0.00125
_C.TRAINING_CONF.SECONDARY_LR = 0.0025
_C.TRAINING_CONF.WEIGHT_L2 = 0.0001
_C.TRAINING_CONF.BIAS_L2 = 0.0
_C.TRAINING_CONF.LR_DECAY_BOUNDARIES = [960000, 1280000]
_C.TRAINING_CONF.LR_DECAY_VALUES = [0.01, 0.001, 0.0001]
_C.TRAINING_CONF.WARMUP_BATCHES = 500
_C.TRAINING_CONF.START_MULTIPLIER = 0.333333333
_C.TRAINING_CONF.MOMENTUM_BETA = 0.9

# ---------------------------------------------------------------------------- #
#  Model Load and Save
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.MODEL_LOAD_PATH = ""
_C.MODEL.MODEL_LOAD_SNAPSHOT_PATH = ""
_C.MODEL.MODEL_SAVE_SNAPSHOT_PATH = ""


def get_default_cfgs():
    """
    Get a yacs CfgNode object with default values for maskrcnn object
    """
    return _C.clone()
