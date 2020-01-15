# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST
from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
#  Env
# ---------------------------------------------------------------------------- #
_C.ENV = CN()
_C.ENV.NUM_GPUS = 4
_C.ENV.IMS_PER_GPU = 2
_C.ENV.ENABLE_INPLACE = False
_C.ENV.CUDNN_BUFFER_SIZE_LIMIT = 1280
_C.ENV.CUDNN_CONV_HEURISTIC_SEARCH_ALGO = True
_C.ENV.CUDNN_CONV_USE_DETERMINISTIC_ALGO_ONLY = False

# ---------------------------------------------------------------------------- #
#  Model art
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
# _C.MODEL.RPN_ONLY = False
# _C.MODEL.MASK_ON = True

_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 600
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 800
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1.0, 1.0, 1.0]

_C.INPUT.MIRROR_PROB = 0.5
_C.INPUT.MAX_BOXES_PER_IMAGE = 128
_C.INPUT.MAX_POLYGONS_PER_OBJECT = 2
_C.INPUT.MAX_POINTS_PER_POLYGON = 256

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = "/dataset/mscoco_2017"
_C.DATASETS.TEST = "/dataset/mscoco_2017"
_C.DATASETS.ANNOTATION_TRAIN = "annotations/instances_train2017.json"
_C.DATASETS.IMAGE_DIR_TRAIN = "train2017"
_C.DATASETS.ANNOTATION_TEST = "annotations/instances_val2017.json"
_C.DATASETS.IMAGE_DIR_TEST = "val2017"
_C.DATASETS.SHUFFLE = True
_C.DATASETS.RANDOM_SEED = 123456
_C.DATASETS.MAX_SEGM_POLY_POINTS_PER_IMAGE = 65536

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 32
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-FPN"
# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0.0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0.0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 1000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 1000

_C.MODEL.RPN.RANDOM_SAMPLE = True
_C.MODEL.RPN.ZERO_CTRL = False

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

_C.MODEL.ROI_HEADS.RANDOM_SAMPLE = True

_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024

# ---------------------------------------------------------------------------- #
#  Mask
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 2
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
# _C.MODEL.ROI_MASK_HEAD.RESOLUTION = 28
_C.MODEL.ROI_MASK_HEAD.ZERO_CTRL = False

# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1

# ---------------------------------------------------------------------------- #
# ResNet50
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()
_C.MODEL.RESNETS.NUM_GROUPS = 1
# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True
_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 180000

_C.SOLVER.MAKE_LR = False
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.OPTIMIZER = "momentum"  # "momentum" or "sgd"
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.STEPS = (120000, 160000)
# _C.SOLVER.GAMMA = 0.1  # GAMMA with 3 step stections -> LR_DECAY_VALUES
_C.SOLVER.LR_DECAY_VALUES = (1, 0.1, 0.01)
_C.SOLVER.ENABLE_LR_DECAY = True

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.ENABLE_WARMUP = True

_C.SOLVER.CHECKPOINT_PERIOD = 0
_C.SOLVER.METRICS_PERIOD = 0

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 8
_C.SOLVER.REDUCE_ALL_LOSSES = False

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #
# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# ---------------------------------------------------------------------------- #
#  Async get oneflow function outputs
# ---------------------------------------------------------------------------- #
_C.ASYNC_GET = True

# ---------------------------------------------------------------------------- #
#  deprecated config
# ---------------------------------------------------------------------------- #
_C.DECODER = CN()
# Anchor generator params
_C.DECODER.FPN_LAYERS = 5
_C.DECODER.FEATURE_MAP_STRIDE = 4
_C.DECODER.ASPECT_RATIOS = [0.5, 1.0, 2.0]
_C.DECODER.ANCHOR_SCALES = 32

# _C.BACKBONE = CN()
# _C.BACKBONE.CONV_BODY = "R-50-FPN"
# _C.BACKBONE.FREEZE_CONV_BODY_AT = 2
# _C.BACKBONE.RESNET_STEM_OUT_CHANNELS = 64

# _C.RPN = CN()
# _C.RPN.RPN_HEAD = "SingleConvRPNHead"

# # Anchor targets params
# _C.RPN.POSITIVE_OVERLAP_THRESHOLD = 0.7
# _C.RPN.NEGATIVE_OVERLAP_THRESHOLD = 0.3
# _C.RPN.SUBSAMPLE_NUM_PER_IMG = 256
# _C.RPN.FOREGROUND_FRACTION = 0.5
# _C.RPN.WEIGHT_X = 1.0
# _C.RPN.WEIGHT_Y = 1.0
# _C.RPN.WEIGHT_W = 1.0
# _C.RPN.WEIGHT_H = 1.0
# _C.RPN.RANDOM_SUBSAMPLE = False

# # RPN post processor
# _C.RPN.TOP_N_PER_FM_TRAIN = 2000
# _C.RPN.NMS_TOP_N_TRAIN = 2000
# _C.RPN.TOP_N_PER_IMG_TRAIN = 1000
# _C.RPN.TOP_N_PER_FM_TEST = 2000
# _C.RPN.NMS_TOP_N_TEST = 2000
# _C.RPN.TOP_N_PER_IMG_TEST = 2000
# _C.RPN.NMS_THRESH = 0.7

# _C.BOX_HEAD = CN()
# _C.BOX_HEAD.FOREGROUND_THRESHOLD = 0.5
# _C.BOX_HEAD.BACKGROUND_THRESHOLD_LOW = 0.0
# _C.BOX_HEAD.BACKGROUND_THRESHOLD_HIGH = 0.5
# _C.BOX_HEAD.FOREGROUND_FRACTION = 0.25
# _C.BOX_HEAD.WEIGHT_X = 10.0
# _C.BOX_HEAD.WEIGHT_Y = 10.0
# _C.BOX_HEAD.WEIGHT_W = 5.0
# _C.BOX_HEAD.WEIGHT_H = 5.0
# _C.BOX_HEAD.NUM_CLASSES = 81
# _C.BOX_HEAD.NUM_SAMPLED_ROI_PER_IMG = 512
# _C.BOX_HEAD.POOLED_H = 7
# _C.BOX_HEAD.POOLED_W = 7
# _C.BOX_HEAD.SPATIAL_SCALE = 0.25
# _C.BOX_HEAD.SAMPLING_RATIO = 2
# _C.BOX_HEAD.RANDOM_SUBSAMPLE = False

# _C.MASK_HEAD = CN()
# _C.MASK_HEAD.POOLED_H = 14
# _C.MASK_HEAD.POOLED_W = 14
# _C.MASK_HEAD.SAMPLING_RATIO = 2
# _C.MASK_HEAD.SPATIAL_SCALE = 0.25


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


def get_sub_path(path, k):
    sub_path = None
    if path == []:
        sub_path = [k]
    else:
        sub_path = path + [k]
    return sub_path


def compare_config(
    d1,
    d2,
    path=None,
    d1_diff=None,
    d2_diff=None,
    d1_only=None,
    d2_only=None,
    identitcal_structure=True,
):
    if d1_diff is None:
        assert path is None
        d1_diff = CN()
    if d2_diff is None:
        assert path is None
        d2_diff = CN()
    if identitcal_structure is False:
        if d1_only is None:
            assert path is None
            d1_only = set()
        if d2_only is None:
            assert path is None
            d2_only = set()
    if path is None:
        path = []
    for k in d1.keys():
        sub_path = get_sub_path(path, k)
        assert sub_path is not None
        if k in d2:
            if type(d1[k]) is CN:
                compare_config(
                    d1[k],
                    d2[k],
                    sub_path,
                    d1_diff,
                    d2_diff,
                    d1_only,
                    d2_only,
                    identitcal_structure=identitcal_structure,
                )
            else:
                if d1[k] != d2[k]:
                    print("{}:".format(".".join(path)))
                    print("- {} : {}".format(k, d1[k]))
                    print("+ {} : {}".format(k, d2[k]))
                    update_by_path(d1_diff, sub_path, d1[k])
                    update_by_path(d2_diff, sub_path, d2[k])
        else:
            if identitcal_structure:
                raise ValueError("key {} not found in d2".format(k))
            else:
                update_by_path(d1_diff, sub_path, d1[k])
                d1_only.add(".".join(sub_path))
    for k_in_d2 in d2.keys():
        sub_path = get_sub_path(path, k_in_d2)
        if k_in_d2 not in d1:
            update_by_path(d2_diff, sub_path, d2[k_in_d2])
            d2_only.add(".".join(sub_path))

    if identitcal_structure:
        return (d1_diff, d2_diff)
    else:
        return (d1_diff, d2_diff, d1_only, d2_only)


def check_compatibility(d1, d2):
    return compare_config(d1, d2, identitcal_structure=False)


if __name__ == "__main__":
    import torch_config
    import argparse

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("-of", "--oneflow_config_yaml", type=str, required=True)
    parser.add_argument("-py", "--pytorch_config_yaml", type=str, required=True)
    args = parser.parse_args()

    torch_cfg = torch_config._C.clone()
    flow_cfg = _C.clone()
    if hasattr(args, "oneflow_config_yaml"):
        flow_cfg.merge_from_file(args.oneflow_config_yaml)
    if hasattr(args, "pytorch_config_yaml"):
        torch_cfg.merge_from_file(args.pytorch_config_yaml)

    (d1_diff, d2_diff, d1_only, d2_only) = check_compatibility(flow_cfg, torch_cfg)
    print("oneflow diff:\n{}\n".format(d1_diff))
    print("pytorch diff:\n{}\n".format(d2_diff))
    print("oneflow only:\n{}\n".format("\n".join(d1_only)))
    print("pytorch only:\n{}\n".format("\n".join(d2_only)))
