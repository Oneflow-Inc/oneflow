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
parser.add_argument("-load", "--model_load_dir", type=str, default="", required=False)
args = parser.parse_args()

# images N,C,H,W
# image_sizes N,2
# gt_boxes N,R,4

@flow.function
def maskrcnn(images, image_sizes, gt_boxes):
    cfg = get_default_cfgs()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)

    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_loss = RPNLoss(cfg)

    image_size_list = flow.piece_slice(image_sizes)
    gt_boxes_list = flow.piece_slice(gt_boxes)

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
    if cfg.TRAINING:
        box_loss, cls_loss = rpn_loss.build(
            anchors, image_size_list, gt_boxes_list, cls_logit_list, bbox_pred_list
        )
    else:
        # TODO: eval net
        pass

    return None

if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.ctrl_port(9788)

    flow.config.default_data_type(flow.float)
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    maskrcnn()
