import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

from backbone import Backbone
from rpn import RPNHead, RPNLoss

from config import get_default_cfgs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", default=None, type=str, required=True, help="yaml config file"
)
args = parser.parse_args()


def maskrcnn():
    cfg = get_default_cfgs()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)

    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_loss = RPNLoss(cfg)

    features = backbone.build(encoded)
    cls_logit_list, bbox_pred_list = rpn_head.build(features)
    if cfg.TRAINING:
        rpn_loss.build(
            anchors, image_size_list, gt_boxes_list, cls_logit_list, bbox_pred_list
        )
    else:
        # TODO: eval net
        pass
