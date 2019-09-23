from config import get_default_cfgs
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

from backbone import Backbone
from rpn import RPNHead, RPNLoss, RPNProposal
from box_head import BoxHead
from mask_head import MaskHead

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    default=None,
    type=str,
    help="yaml config file",
)
parser.add_argument(
    "-load", "--model_load_dir", type=str, default="", required=False
)
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)

args = parser.parse_args()

def get_numpy_placeholders():
    import numpy as np
    (N,C,H,W) = (2,3,256,256)
    R=50
    G=12
    return (
        (
            np.random.randn(N,C,H,W).astype(np.float32),
            np.random.randn(N,2).astype(np.float32),
            np.random.randn(N,R,4).astype(np.float32),
            np.random.randn(N,G).astype(np.int32),
        ),
        (
            flow.input_blob_def((N,C,H,W)),
            flow.input_blob_def((N, 2)),
            flow.input_blob_def((N,R,4)),
            flow.input_blob_def((N,G)),
        )
    )
placeholders = get_numpy_placeholders()
@flow.function
def maskrcnn(images=placeholders[1][0], image_sizes=placeholders[1][1], gt_boxes=placeholders[1][2], gt_labels=placeholders[1][3]):
# def maskrcnn(images, image_sizes, gt_boxes, gt_segms, gt_labels):
    r"""Mask-RCNN
    Args:
    images: N,C,H,W
    image_sizes: N,2
    gt_boxes: N,R,4
    gt_labels: N,G
    """
    cfg = get_default_cfgs()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)
    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_loss = RPNLoss(cfg)
    rpn_proposal = RPNProposal(cfg)
    box_head = BoxHead(cfg)
    mask_head = MaskHead(cfg)

    image_size_list = flow.piece_slice(image_sizes)
    gt_boxes_list = flow.piece_slice(gt_boxes)
    gt_labels_list = flow.piece_slice(gt_labels)
    gt_segms_list = flow.piece_slice(gt_segms)

    anchors = []
    for i in range(cfg.DECODER.FPN_LAYERS):
        anchors.append(
            flow.detection.anchor_generate(
                images=images,
                feature_map_stride=cfg.DECODER.FEATURE_MAP_STRIDE * pow(2, i),
                aspect_ratios=cfg.DECODER.ASPECT_RATIOS,
                anchor_scales=cfg.DECODER.ANCHOR_SCALES * pow(2, i),
            )
        )

    # Backbone
    features = backbone.build(images)

    # RPN
    cls_logit_list, bbox_pred_list = rpn_head.build(features)
    rpn_bbox_loss, rpn_objectness_loss = rpn_loss.build(
        anchors, image_size_list, gt_boxes_list, bbox_pred_list, cls_logit_list
    )
    # proposals = rpn_proposal.build(
    #     anchors, cls_logit_list, bbox_pred_list, image_size_list, gt_boxes_list
    # )

    # Box Head
    # box_loss, cls_loss, pos_proposal_list, pos_gt_indices_list = box_head.build_train(
    #     proposals, gt_boxes_list, gt_labels_list, features
    # )

    # Mask Head
    # mask_loss = mask_head.build_train(
    #     pos_proposal_list, pos_gt_indices_list, gt_segms_list, gt_labels_list, features
    # )

    # return rpn_bbox_loss, rpn_objectness_loss, box_loss, cls_loss, mask_loss

    return rpn_bbox_loss, rpn_objectness_loss


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.ctrl_port(9788)

    flow.config.default_data_type(flow.float)
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    maskrcnn(images=placeholders[0][0], image_sizes=placeholders[0][1], gt_boxes=placeholders[0][2], gt_labels=placeholders[0][3])
