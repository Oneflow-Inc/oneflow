# must import get cfg before importing oneflow
from config import get_default_cfgs

import os
import numpy as np
import argparse
import oneflow as flow

from datetime import datetime
from backbone import Backbone
from rpn import RPNHead, RPNLoss, RPNProposal
from box_head import BoxHead
from mask_head import MaskHead

from eval.bounding_box import BoxList
from eval.box_head_inference import PostProcessor
from eval.mask_head_inference import MaskPostProcessor
from eval.coco import COCODataset
from eval.coco_eval import do_coco_evaluation


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", default=None, type=str, help="yaml config file")
parser.add_argument("-bz", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default="", required=False)
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-cp", "--ctrl_port", type=int, default=19765, required=False)
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
terminal_args = parser.parse_args()


@flow.function
def maskrcnn_box_head_eval_job(
    images=flow.input_blob_def((2, 3, 416, 608), dtype=flow.float, is_dynamic=True),
    image_sizes=flow.input_blob_def((2, 2), dtype=flow.int32, is_dynamic=False),
):
    cfg = get_default_cfgs()
    if terminal_args.config_file is not None:
        cfg.merge_from_file(terminal_args.config_file)
    cfg.freeze()
    print(cfg)
    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_proposal = RPNProposal(cfg)
    box_head = BoxHead(cfg)

    image_size_list = [
        flow.squeeze(flow.local_gather(image_sizes, flow.constant(i, dtype=flow.int32)), [0])
        for i in range(image_sizes.shape[0])
    ]
    anchors = []
    for i in range(cfg.DECODER.FPN_LAYERS):
        anchors.append(
            flow.detection.anchor_generate(
                images=flow.transpose(images, perm=[0, 2, 3, 1]),
                feature_map_stride=cfg.DECODER.FEATURE_MAP_STRIDE * pow(2, i),
                aspect_ratios=cfg.DECODER.ASPECT_RATIOS,
                anchor_scales=cfg.DECODER.ANCHOR_SCALES * pow(2, i),
            )
        )

    # Backbone
    features = backbone.build(images)

    # RPN
    cls_logit_list, bbox_pred_list = rpn_head.build(features)
    proposals = rpn_proposal.build(anchors, cls_logit_list, bbox_pred_list, image_size_list, None)

    # Box Head
    cls_probs, box_regressions = box_head.build_eval(proposals, features)

    return tuple(proposals) + tuple(features) + (cls_probs,) + (box_regressions,)


@flow.function
def maskrcnn_mask_head_eval_job(
    detection0=flow.input_blob_def((1000, 4), dtype=flow.float, is_dynamic=True),
    detection1=flow.input_blob_def((1000, 4), dtype=flow.float, is_dynamic=True),
    fpn_fm1=flow.input_blob_def((2, 256, 104, 152), dtype=flow.float, is_dynamic=True),
    fpn_fm2=flow.input_blob_def((2, 256, 52, 76), dtype=flow.float, is_dynamic=True),
    fpn_fm3=flow.input_blob_def((2, 256, 26, 38), dtype=flow.float, is_dynamic=True),
    fpn_fm4=flow.input_blob_def((2, 256, 13, 19), dtype=flow.float, is_dynamic=True),
):
    cfg = get_default_cfgs()
    mask_head = MaskHead(cfg)
    mask_logits = mask_head.build_eval(
        [detection0, detection1], [fpn_fm1, fpn_fm2, fpn_fm3, fpn_fm4]
    )
    mask_prob = flow.math.sigmoid(mask_logits)
    return mask_prob


if __name__ == "__main__":
    flow.config.gpu_device_num(terminal_args.gpu_num_per_node)
    flow.config.ctrl_port(terminal_args.ctrl_port)
    flow.config.default_data_type(flow.float)

    assert terminal_args.model_load_dir != ""
    check_point = flow.train.CheckPoint()
    check_point.load(terminal_args.model_load_dir)

    images = np.load("/tmp/shared_with_jxf/maskrcnn_eval_input_data_small/images.npy")
    image_sizes = np.load("/tmp/shared_with_jxf/maskrcnn_eval_input_data_small/image_sizes.npy")
    image_num = image_sizes.shape[0]

    # Box Head and Post-Processor
    results = maskrcnn_box_head_eval_job(images, image_sizes).get()
    cls_probs = results[-2]
    box_regressions = results[-1]
    fpn_feature_map = []
    for i in range(4):
        fpn_feature_map.append(results[image_num + i].ndarray())
    boxes = []
    for proposal, img_size in zip(results[:image_num], image_sizes):
        bbox = BoxList(proposal.ndarray(), (img_size[1], img_size[0]), mode="xyxy")
        boxes.append(bbox)
    postprocessor = PostProcessor()
    results = postprocessor.forward((cls_probs.ndarray(), box_regressions.ndarray()), boxes)

    # Mask Head and Post-Processor
    detections = []
    for result in results:
        detections.append(result.bbox)
    mask_prob = maskrcnn_mask_head_eval_job(
        detections[0],
        detections[1],
        fpn_feature_map[0],
        fpn_feature_map[1],
        fpn_feature_map[2],
        fpn_feature_map[3],
    ).get()
    mask_postprocessor = MaskPostProcessor()
    predictions = mask_postprocessor.forward(mask_prob.ndarray(), results)

    # Calculate mAP
    ann_file = "/dataset/mscoco_2017/annotations/sample_2_instances_val2017.json"
    dataset = COCODataset(ann_file)
    do_coco_evaluation(
        dataset,
        predictions,
        box_only=False,
        output_folder="./output",
        iou_types=["bbox", "segm"],
        expected_results=(),
        expected_results_sigma_tol=4,
    )
