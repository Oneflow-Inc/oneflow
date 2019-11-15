# must import get cfg before importing oneflow
from config import get_default_cfgs

import os
import numpy as np
import pickle
import argparse
import oneflow as flow
import oneflow.core.data.data_pb2 as data_util

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
parser.add_argument(
    "-dataset_dir", "--dataset_dir", type=str, default="/dataset/mscoco_2017", required=False
)
parser.add_argument(
    "-anno", "--annotation_file", type=str, default="instances_val2017.json", required=False
)
parser.add_argument("-imgd", "--image_dir", type=str, default="val2017", required=False)
parser.add_argument("-fake", "--fake_img", default=False, action="store_true", required=False)
terminal_args = parser.parse_args()


def make_data_loader(
    batch_size,
    batch_cache_size=3,
    dataset_dir="/dataset/mscoco_2017",
    annotation_file="annotations/sample_2_instances_val2017.json",
    image_dir="val2017",
    random_seed=873898,
    shuffle=False,
    group_by_aspect_ratio=True,
):
    coco = flow.data.COCODataset(
        dataset_dir, annotation_file, image_dir, random_seed, shuffle, group_by_aspect_ratio
    )
    data_loader = flow.data.DataLoader(coco, batch_size, batch_cache_size)
    data_loader.add_blob(
        "image",
        data_util.DataSourceCase.kImage,
        shape=(416, 608, 3),
        dtype=flow.float,
        is_dynamic=True,
    )
    data_loader.add_blob(
        "image_size", data_util.DataSourceCase.kImageSize, shape=(2,), dtype=flow.int32
    )
    data_loader.add_blob(
        "image_id", data_util.DataSourceCase.kImageId, shape=(1,), dtype=flow.int64
    )
    data_loader.add_transform(flow.data.TargetResizeTransform(400, 600))
    data_loader.add_transform(flow.data.ImageNormalizeByChannel((102.9801, 115.9465, 122.7717)))
    data_loader.add_transform(flow.data.ImageAlign(32))
    data_loader.init()
    return data_loader


def maskrcnn_box_head_eval(images, image_sizes):
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
                images=images,
                feature_map_stride=cfg.DECODER.FEATURE_MAP_STRIDE * pow(2, i),
                aspect_ratios=cfg.DECODER.ASPECT_RATIOS,
                anchor_scales=cfg.DECODER.ANCHOR_SCALES * pow(2, i),
            )
        )

    # Backbone
    features = backbone.build(flow.transpose(images, perm=[0, 3, 1, 2]))

    # RPN
    cls_logit_list, bbox_pred_list = rpn_head.build(features)
    proposals = rpn_proposal.build(anchors, cls_logit_list, bbox_pred_list, image_size_list, None)

    # Box Head
    cls_probs, box_regressions = box_head.build_eval(proposals, features)

    return tuple(proposals) + tuple(features) + (image_sizes,) + (cls_probs,) + (box_regressions,)


if terminal_args.fake_img:

    @flow.function
    def maskrcnn_box_head_eval_job(
        images=flow.input_blob_def(shape=(2, 3, 416, 608), dtype=flow.float32, is_dynamic=True)
    ):
        data_loader = make_data_loader(
            batch_size=terminal_args.batch_size,
            batch_cache_size=3,
            dataset_dir=terminal_args.dataset_dir,
            annotation_file=terminal_args.annotation_file,
            image_dir=terminal_args.image_dir,
        )
        image_sizes = data_loader("image_size")
        image_ids = data_loader("image_id")

        image_trans = flow.transpose(images, perm=[0, 2, 3, 1])
        return maskrcnn_box_head_eval(image_trans, image_sizes) + (image_ids,)


else:

    @flow.function
    def maskrcnn_box_head_eval_job():
        data_loader = make_data_loader(
            batch_size=terminal_args.batch_size,
            batch_cache_size=3,
            dataset_dir=terminal_args.dataset_dir,
            annotation_file=terminal_args.annotation_file,
            image_dir=terminal_args.image_dir,
        )
        images = data_loader("image")
        image_sizes = data_loader("image_size")
        image_ids = data_loader("image_id")

        return maskrcnn_box_head_eval(images, image_sizes) + (image_ids,)


def parse_results_from_box_head(results):
    image_sizes = results[-4]
    cls_probs = results[-3]
    box_regressions = results[-2]
    image_ids = results[-1]
    image_num = image_sizes.shape[0]
    feature_maps = []
    for i in range(4):
        feature_maps.append(results[image_num + i].ndarray())
    box_lists = []
    for proposal, img_size in zip(results[:image_num], image_sizes):
        box_list = BoxList(proposal.ndarray(), (img_size[1], img_size[0]), mode="xyxy")
        box_lists.append(box_list)

    return_dict = {
        "cls_probs": cls_probs,
        "box_regressions": box_regressions,
        "box_lists": box_lists,
        "feature_maps": feature_maps,
        "image_ids": image_ids,
    }

    return return_dict


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
    flow.config.cudnn_conv_heuristic_search_algo(True)
    flow.config.cudnn_conv_use_deterministic_algo_only(False)

    assert terminal_args.model_load_dir != ""
    check_point = flow.train.CheckPoint()
    check_point.load(terminal_args.model_load_dir)

    prediction_list = []
    image_id_list = []
    for i in range(terminal_args.iter_num):
        # Box Head
        if terminal_args.fake_img:
            if i == 0:
                f = open("/dataset/mask_rcnn/maskrcnn_eval_net_10/fake_image_list.pkl", "rb")
                fake_image_list = pickle.load(f)
            results = maskrcnn_box_head_eval_job(fake_image_list[i]).get()
        else:
            results = maskrcnn_box_head_eval_job().get()
        # We have to write such ugly parsing code because oneflow job can only return list or tuple of blob
        return_dict = parse_results_from_box_head(results)
        cls_probs = return_dict["cls_probs"]
        box_regressions = return_dict["box_regressions"]
        box_lists = return_dict["box_lists"]
        feature_maps = return_dict["feature_maps"]
        image_ids = return_dict["image_ids"]

        # Box Head Post-Processor
        postprocessor = PostProcessor()
        box_head_predictions = postprocessor.forward(
            (cls_probs.ndarray(), box_regressions.ndarray()), box_lists
        )

        # Mask Head and Post-Processor
        detections = []
        for prediction in box_head_predictions:
            detections.append(prediction.bbox)
        mask_prob = maskrcnn_mask_head_eval_job(
            detections[0],
            detections[1],
            feature_maps[0],
            feature_maps[1],
            feature_maps[2],
            feature_maps[3],
        ).get()

        # Mask Head Post-Processor
        mask_postprocessor = MaskPostProcessor()
        predictions = mask_postprocessor.forward(mask_prob.ndarray(), box_head_predictions)

        image_id_list += list(np.squeeze(image_ids.ndarray()))
        prediction_list += predictions

    # Sort predictions by image_id
    num_imgs = len(prediction_list)
    assert num_imgs == len(image_id_list)
    prediction_dict = {}
    for i in range(num_imgs):
        prediction_dict.update({image_id_list[i]: prediction_list[i]})
    sorted_image_ids = list(sorted(prediction_dict.keys()))
    sorted_predictions = [prediction_dict[i] for i in sorted_image_ids]

    # Calculate mAP
    ann_file = os.path.join(terminal_args.dataset_dir, terminal_args.annotation_file)
    dataset = COCODataset(ann_file)
    do_coco_evaluation(
        dataset,
        sorted_predictions,
        box_only=False,
        output_folder="./output",
        iou_types=["bbox", "segm"],
        expected_results=(),
        expected_results_sigma_tol=4,
    )
