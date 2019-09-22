import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

from backbone import Backbone
from rpn import RPNHead, RPNLoss, RPNProposal

from config import get_default_cfgs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", default=None, type=str, required=True, help="yaml config file"
)
args = parser.parse_args()


def maskrcnn(images, image_sizes, gt_boxes, gt_labels):
    cfg = get_default_cfgs()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)

    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_loss = RPNLoss(cfg)
    rpn_proposal = RPNProposal(cfg)
    box_head = BoxHead(cfg)

    image_size_list = flow.piece_slice(image_sizes)
    gt_boxes_list = flow.piece_slice(gt_boxes)
    gt_labels_list = flow.piece_slice(gt_labels)

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
    proposals = rpn_proposal.build(
        anchors,
        cls_logit_list,
        bbox_pred_list,
        image_size_list,
        gt_boxes_list,
    )

    # Box Head
    box_head.build_train(proposals, gt_boxes_list, gt_labels_list, features)

    return rpn_bbox_loss, rpn_objectness_loss

