# must import get cfg before importing oneflow
from config import get_default_cfgs

import os
import numpy as np
import pickle
import argparse
from datetime import datetime

import oneflow as flow
import oneflow.core.data.data_pb2 as data_util
from distribution import distribute_execute
from backbone import Backbone
from rpn import RPNHead, RPNLoss, RPNProposal
from box_head import BoxHead
from mask_head import MaskHead
from eval.bounding_box import BoxList
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


@flow.experimental.mirror_execute(terminal_args.gpu_num_per_node, 1)
def maskrcnn_eval(images, image_sizes, image_ids):
    cfg = get_default_cfgs()
    if terminal_args.config_file is not None:
        cfg.merge_from_file(terminal_args.config_file)
    cfg.freeze()
    print(cfg)
    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_proposal = RPNProposal(cfg)
    box_head = BoxHead(cfg)
    mask_head = MaskHead(cfg)

    assert images.shape[3] == 3
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
    box_head_results = box_head.build_eval(proposals, features, image_size_list)

    # Mask Head
    detections = [result[0] for result in box_head_results]
    mask_prob = mask_head.build_eval(detections, features)

    return {"box_head_results": box_head_results, "mask_prob": mask_prob, "image_ids": image_ids}


if terminal_args.fake_img:

    @flow.function
    def maskrcnn_eval_job(
        images=flow.input_blob_def(shape=(2, 416, 608, 3), dtype=flow.float32, is_dynamic=True)
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

        return maskrcnn_eval(images, image_sizes, image_ids)


else:

    @flow.function
    def maskrcnn_eval_job():
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

        return maskrcnn_eval(images, image_sizes, image_ids)


# return: (list of BoxList wrt. images, list of image_is wrt. image)
def GenPredictionsAndImageIds(results):
    assert isinstance(results, dict)
    box_head_results = results["box_head_results"]
    mask_prob = results["mask_prob"]
    image_ids = results["image_ids"]

    boxes = []
    for box_head_result in box_head_results:
        # swap height and width in image_size
        image_size_h_w = box_head_result[-1]
        assert len(image_size_h_w) == 2
        image_size_w_h = [image_size_h_w[1], image_size_h_w[0]]
        boxlist = BoxList(box_head_result[0].ndarray(), image_size_w_h, mode="xyxy")
        boxlist.add_field("scores", box_head_result[1].ndarray())
        boxlist.add_field("labels", box_head_result[2].ndarray())
        boxes.append(boxlist)

    # Mask Head Post-Processor
    mask_postprocessor = MaskPostProcessor()
    predictions = mask_postprocessor.forward(mask_prob.ndarray(), boxes)
    image_ids = list(np.squeeze(image_ids.ndarray(), axis=1))

    return (predictions, image_ids)


if __name__ == "__main__":
    flow.config.gpu_device_num(terminal_args.gpu_num_per_node)
    flow.env.ctrl_port(terminal_args.ctrl_port)
    flow.config.default_data_type(flow.float)
    flow.config.cudnn_conv_heuristic_search_algo(True)
    flow.config.cudnn_conv_use_deterministic_algo_only(False)

    assert terminal_args.model_load_dir != ""
    check_point = flow.train.CheckPoint()
    check_point.load(terminal_args.model_load_dir)

    predictions_all = []
    image_ids_all = []
    for i in range(terminal_args.iter_num):  # wrt. iterations
        if terminal_args.fake_img:
            if i == 0:
                f = open("/dataset/mask_rcnn/maskrcnn_eval_net_10/fake_image_list.pkl", "rb")
                fake_image_list = pickle.load(f)
            images = fake_image_list[i].transpose((0, 2, 3, 1)).copy()
            results = maskrcnn_eval_job(images).get()
        else:
            results = maskrcnn_eval_job().get()

        for device_id in range(terminal_args.gpu_num_per_node):  # wrt. devices
            predictions_per_gpu, image_ids_per_gpu = GenPredictionsAndImageIds(results[device_id])
            predictions_all += predictions_per_gpu
            image_ids_all += image_ids_per_gpu

    # Sort predictions by image_id
    num_imgs = len(predictions_all)
    assert num_imgs == len(image_ids_all)
    prediction_dict = {}
    for i in range(num_imgs):
        prediction_dict.update({image_ids_all[i]: predictions_all[i]})
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

# Box Head
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.783
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.550
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.592
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.490
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.590
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.625

# Mask Head
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.750
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.650
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.667
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.490
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.580
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.580
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.667
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450
