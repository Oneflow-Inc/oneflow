# must import get cfg before importing oneflow
from config import get_default_cfgs

import os
import numpy as np
import pandas as pd
import argparse
import time
import statistics
import oneflow as flow

from datetime import datetime
from backbone import Backbone
from rpn import RPNHead, RPNLoss, RPNProposal
from box_head import BoxHead
from mask_head import MaskHead
from data_load import make_data_loader
from blob_watcher import save_blob_watched, blob_watched, diff_blob_watched


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument(
    "-c", "--config_file", help="yaml config file", type=str, required=False
)
parser.add_argument(
    "-cp", "--ctrl_port", type=int, default=19765, required=False
)
parser.add_argument(
    "-g", "--gpu_num_per_node", type=int, default=1, required=False
)
parser.add_argument("-bz", "--batch_size", type=int, default=2, required=False)
parser.add_argument("-m", "--model_dir", type=str, required=False)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="./model_save-{}".format(
        str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    ),
    required=False,
)
parser.add_argument(
    "--save_model_every_n_batch", type=int, default=0, required=False
)
parser.add_argument(
    "--save_loss_npy_every_n_batch", type=int, default=0, required=False
)
parser.add_argument("-data", "--dataset_dir", type=str, required=False)
parser.add_argument("-anno", "--annotation_file", type=str, required=False)
parser.add_argument("-img", "--image_dir", type=str, required=False)
parser.add_argument("-fake", "--fake_image_path", type=str, required=False)
parser.add_argument("-lr", "--primary_lr", type=float, required=False)
parser.add_argument("-slr", "--secondary_lr", type=float, required=False)
parser.add_argument("--enable_warmup", type=str2bool, nargs="?", const=True)
parser.add_argument("--enable_lr_decay", type=str2bool, nargs="?", const=True)
parser.add_argument("--shuffle_dataset", type=str2bool, nargs="?", const=True)
parser.add_argument(
    "--rpn_random_sample", type=str2bool, nargs="?", const=True
)
parser.add_argument(
    "--roi_head_random_sample", type=str2bool, nargs="?", const=True
)
parser.add_argument(
    "-flip", "--random_flip_image", type=str2bool, nargs="?", const=True
)
parser.add_argument(
    "-pr",
    "--print_loss_each_rank",
    default=False,
    action="store_true",
    required=False,
)
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument(
    "-j", "--jupyter", default=False, action="store_true", required=False
)
parser.add_argument(
    "-v", "--verbose", default=False, action="store_true", required=False
)

terminal_args = parser.parse_args()


def MakeWatcherCallback(prompt):
    def Callback(blob, blob_def):
        if prompt == "forward":
            return
        print(
            "%s, lbn: %s, min: %s, max: %s"
            % (prompt, blob_def.logical_blob_name, blob.min(), blob.max())
        )

    return Callback


def maskrcnn_train(cfg, image, image_size, gt_bbox, gt_segm, gt_label):
    """Mask-RCNN
    Args:
        image: (N, H, W, C)
        image_size: (N, 2)
        gt_bbox: (N, M, 4), num_lod_lvl == 2
        gt_segm: (N, M, 28, 28), num_lod_lvl == 2
        gt_label: (N, M), num_lod_lvl == 2
    """
    assert image.shape[3] == 3
    assert gt_bbox.num_of_lod_levels == 2
    assert gt_segm.num_of_lod_levels == 2
    assert gt_label.num_of_lod_levels == 2

    backbone = Backbone(cfg)
    rpn_head = RPNHead(cfg)
    rpn_loss = RPNLoss(cfg)
    rpn_proposal = RPNProposal(cfg)
    box_head = BoxHead(cfg)
    mask_head = MaskHead(cfg)

    image_size_list = [
        flow.squeeze(
            flow.local_gather(image_size, flow.constant(i, dtype=flow.int32)),
            [0],
            name="image{}_size".format(i),
        )
        for i in range(image_size.shape[0])
    ]

    gt_bbox_list = flow.piece_slice(
        gt_bbox, gt_bbox.shape[0], name="gt_bbox_per_img"
    )

    gt_label_list = flow.piece_slice(
        gt_label, gt_label.shape[0], name="gt_label_per_img"
    )

    gt_segm_list = flow.piece_slice(
        gt_segm, gt_segm.shape[0], name="gt_segm_per_img"
    )

    anchors = [
        flow.detection.anchor_generate(
            images=image,
            feature_map_stride=cfg.DECODER.FEATURE_MAP_STRIDE * pow(2, i),
            aspect_ratios=cfg.DECODER.ASPECT_RATIOS,
            anchor_scales=cfg.DECODER.ANCHOR_SCALES * pow(2, i),
        )
        for i in range(cfg.DECODER.FPN_LAYERS)
    ]

    # Backbone
    # CHECK_POINT: fpn features
    # with flow.watch_scope(
    #     blob_watcher=blob_watched, diff_blob_watcher=diff_blob_watched
    # ):
    features = backbone.build(flow.transpose(image, perm=[0, 3, 1, 2]))

    # RPN
    # with flow.watch_scope(
    #     blob_watcher=blob_watched, diff_blob_watcher=diff_blob_watched
    # ):
    cls_logit_list, bbox_pred_list = rpn_head.build(features)
    rpn_bbox_loss, rpn_objectness_loss = rpn_loss.build(
        anchors, image_size_list, gt_bbox_list, bbox_pred_list, cls_logit_list
    )

    # with flow.watch_scope(blob_watched):
    proposals = rpn_proposal.build(
        anchors, cls_logit_list, bbox_pred_list, image_size_list, gt_bbox_list
    )

    # with flow.watch_scope(
    #     blob_watcher=blob_watched, diff_blob_watcher=diff_blob_watched
    # ), flow.watch_scope(
    #     blob_watcher=MakeWatcherCallback("forward"),
    #     diff_blob_watcher=MakeWatcherCallback("backward"),
    # ):
    # Box Head
    box_loss, cls_loss, pos_proposal_list, pos_gt_indices_list, total_pos_inds_elem_cnt = box_head.build_train(
        proposals, gt_bbox_list, gt_label_list, features
    )

    # Mask Head
    mask_loss = mask_head.build_train(
        pos_proposal_list,
        pos_gt_indices_list,
        gt_segm_list,
        gt_label_list,
        features,
    )

    return {
        "loss_rpn_box_reg": rpn_bbox_loss,
        "loss_objectness": rpn_objectness_loss,
        "loss_box_reg": box_loss,
        "loss_classifier": cls_loss,
        "loss_mask": mask_loss,
        "total_pos_inds_elem_cnt": total_pos_inds_elem_cnt,
    }


def compare_config(d1, d2, path=""):
    from yacs.config import CfgNode as CN
    for k in d1.keys():
        if not d2.has_key(k):
            raise ValueError
        else:
            if type(d1[k]) is CN:
                sub_path = None
                if path == "":
                    sub_path = k
                else:
                    sub_path = path + "." + k
                assert sub_path is not ""
                compare_config(d1[k],d2[k], sub_path)
            else:
                if d1[k] != d2[k]:
                    print path, ":"
                    print " - ", k," : ", d1[k]
                    print " + ", k," : ", d2[k] 

def merge_args_to_train_config(args):
    config = get_default_cfgs()

    if hasattr(args, "config_file"):
        config.merge_from_file(args.config_file)
        print("merged config from {}".format(args.config_file))

    if hasattr(args, "gpu_num_per_node"):
        config.ENV.NUM_GPUS = args.gpu_num_per_node

    if hasattr(args, "batch_size"):
        config.TRAIN.BATCH_SIZE = args.batch_size

    assert config.TRAIN.BATCH_SIZE % config.ENV.NUM_GPUS == 0
    config.TRAIN.IMAGE_PER_GPU = int(
        config.TRAIN.BATCH_SIZE / config.ENV.NUM_GPUS
    )

    if hasattr(args, "model_dir"):
        config.TRAIN.MODEL_INIT_PATH = args.model_dir

    if hasattr(args, "dataset_dir"):
        config.TRAIN.DATASET.DATASET_DIR = args.dataset_dir

    if hasattr(args, "annotation_file"):
        config.TRAIN.DATASET.ANNOTATION = args.annotation_file

    if hasattr(args, "image_dir"):
        config.TRAIN.DATASET.IMAGE_DIR = args.image_dir

    if hasattr(args, "primary_lr"):
        config.SOLVER.PRIMARY_LR = args.primary_lr

    if hasattr(args, "secondary_lr"):
        config.SOLVER.SECONDARY_LR = args.secondary_lr

    if hasattr(args, "enable_warmup"):
        config.SOLVER.ENABLE_WARMUP = args.enable_warmup

    if hasattr(args, "enable_lr_decay"):
        config.SOLVER.ENABLE_LR_DECAY = args.enable_lr_decay

    if hasattr(args, "shuffle_dataset"):
        config.TRAIN.DATASET.SHUFFLE = args.shuffle_dataset

    if hasattr(args, "rpn_random_sample"):
        config.DEBUG.RPN_RANDOM_SAMPLE = args.rpn_random_sample

    if hasattr(args, "roi_head_random_sample"):
        config.DEBUG.ROI_HEAD_RANDOM_SAMPLE = args.roi_head_random_sample

    if hasattr(args, "random_flip_image"):
        if args.random_flip_image is False:
            config.TRAIN.INPUT.MIRROR_PROB = 0.0

    config.freeze()
    print("difference between default (upper) and given config (lower)")
    compare_config(get_default_cfgs(), config)
    if args.verbose:
        print(config)

    return config


def set_config(cfg):
    flow.config.cudnn_buf_limit_mbyte(cfg.ENV.CUDNN_BUFFER_SIZE_LIMIT)
    flow.config.cudnn_conv_heuristic_search_algo(
        cfg.ENV.CUDNN_CONV_HEURISTIC_SEARCH_ALGO
    )
    flow.config.cudnn_conv_use_deterministic_algo_only(
        cfg.ENV.CUDNN_CONV_USE_DETERMINISTIC_ALGO_ONLY
    )
    flow.config.default_initialize_with_snapshot_path(cfg.TRAIN.MODEL_INIT_PATH)
    flow.config.train.primary_lr(cfg.SOLVER.PRIMARY_LR)
    flow.config.train.secondary_lr(cfg.SOLVER.SECONDARY_LR)
    flow.config.train.weight_l2(cfg.SOLVER.WEIGHT_L2)
    flow.config.train.bias_l2(cfg.SOLVER.BIAS_L2)

    if cfg.SOLVER.OPTIMIZER == "momentum":
        optimizer = dict(momentum_conf={"beta": cfg.SOLVER.MOMENTUM_BETA})
    elif cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = dict(naive_conf={})
    else:
        raise ValueError("optimizer must be 'momentum' or 'sgd'")

    if cfg.SOLVER.ENABLE_LR_DECAY:
        optimizer.update(
            dict(
                learning_rate_decay=dict(
                    piecewise_constant_conf={
                        "boundaries": cfg.SOLVER.LR_DECAY_BOUNDARIES,
                        "values": cfg.SOLVER.LR_DECAY_VALUES,
                    }
                )
            )
        )

    if cfg.SOLVER.ENABLE_WARMUP:
        if cfg.SOLVER.WARMUP_METHOD == "linear":
            optimizer.update(
                {
                    "warmup_conf": {
                        "linear_conf": {
                            "warmup_batches": cfg.SOLVER.WARMUP_BATCHES,
                            "start_multiplier": cfg.SOLVER.WARMUP_FACTOR,
                        }
                    }
                }
            )
        elif cfg.SOLVER.WARMUP_METHOD == "constant":
            raise NotImplementedError
        else:
            raise ValueError("warmup method must be 'linear' or 'constant'")

    flow.config.train.model_update_conf(optimizer)


def save_model(check_point, i):
    if not os.path.exists(terminal_args.model_save_dir):
        os.makedirs(terminal_args.model_save_dir)
    model_dst = os.path.join(terminal_args.model_save_dir, "iter-" + str(i))
    print("saving models to {}".format(model_dst))
    check_point.save(model_dst)


def train_net(config, image=None):
    data_loader = make_data_loader(config, True)

    loss_name_tup = (
        "loss_rpn_box_reg",
        "loss_objectness",
        "loss_box_reg",
        "loss_classifier",
        "loss_mask",
    )

    if config.ENV.NUM_GPUS > 1:
        distribute_train_func = flow.experimental.mirror_execute(
            config.ENV.NUM_GPUS, 1
        )(maskrcnn_train)
        outputs = distribute_train_func(
            config,
            flow.identity(image) if image else data_loader("image"),
            data_loader("image_size"),
            data_loader("gt_bbox"),
            data_loader("gt_segm"),
            data_loader("gt_labels"),
        )
        for outputs_per_rank in outputs:
            for k, v in outputs_per_rank.items():
                if k in loss_name_tup:
                    flow.losses.add_loss(v)

    else:
        outputs = maskrcnn_train(
            config,
            image or data_loader("image"),
            data_loader("image_size"),
            data_loader("gt_bbox"),
            data_loader("gt_segm"),
            data_loader("gt_labels"),
        )
        for k, v in outputs.items():
            if k in loss_name_tup:
                flow.losses.add_loss(v)

    return outputs


def init_train_func(config, input_fake_image):
    flow.env.ctrl_port(terminal_args.ctrl_port)
    flow.config.enable_inplace(config.ENV.ENABLE_INPLACE)
    flow.config.gpu_device_num(config.ENV.NUM_GPUS)
    flow.config.default_data_type(flow.float)

    if input_fake_image:

        @flow.function
        def train(
            image_blob=flow.input_blob_def(
                shape=(2, 800, 1344, 3), dtype=flow.float32, is_dynamic=True
            )
        ):
            set_config(config)
            return train_net(config, image_blob)

        return train

    else:

        @flow.function
        def train():
            set_config(config)
            return train_net(config)

        return train


def print_metric_title():
    if terminal_args.print_loss_each_rank:
        print(
            "{:<8} {:<8} {:<16} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
                "iter",
                "rank",
                "elapsed_time",
                "loss_rpn_box_reg",
                "loss_objectness",
                "loss_box_reg",
                "loss_classifier",
                "loss_mask",
            )
        )
    else:
        print(
            "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
                "iter",
                "elapsed_time",
                "loss_rpn_box_reg",
                "loss_objectness",
                "loss_box_reg",
                "loss_classifier",
                "loss_mask",
            )
        )


def update_metrics(metrics, iter, elapsed_time, outputs):
    elapsed_time_str = "{:.6f}".format(elapsed_time)
    if terminal_args.print_loss_each_rank:
        for rank, output_per_rank in enumerate(outputs):
            fmt = "{:<8} {:<8} {:<16} {:<16.8f} {:<16.8f} {:<16.8f} {:<16.8f} {:<16.8f} {:<8}"
            loss_rpn_box_reg = output_per_rank["loss_rpn_box_reg"]
            loss_objectness = output_per_rank["loss_objectness"]
            loss_box_reg = output_per_rank["loss_box_reg"]
            loss_classifier = output_per_rank["loss_classifier"]
            loss_mask = output_per_rank["loss_mask"]
            total_pos_inds_elem_cnt = output_per_rank["total_pos_inds_elem_cnt"]
            print(
                fmt.format(
                    iter,
                    rank,
                    elapsed_time_str if rank == 0 else "",
                    loss_rpn_box_reg.mean(),
                    loss_objectness.mean(),
                    loss_box_reg.mean(),
                    loss_classifier.mean(),
                    loss_mask.mean(),
                    int(total_pos_inds_elem_cnt.item()),
                )
            )
    else:

        def reduce_across_ranks(losses):
            loss_list_of_ranks = [loss.mean() for loss in losses]
            return sum(loss_list_of_ranks) / len(loss_list_of_ranks)

        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        loss_rpn_box_reg = reduce_across_ranks(
            [output["loss_rpn_box_reg"] for output in outputs]
        )
        loss_objectness = reduce_across_ranks(
            [output["loss_objectness"] for output in outputs]
        )
        loss_box_reg = reduce_across_ranks(
            [output["loss_box_reg"] for output in outputs]
        )
        loss_classifier = reduce_across_ranks(
            [output["loss_classifier"] for output in outputs]
        )
        loss_mask = reduce_across_ranks([output["loss_mask"] for output in outputs])
        total_pos_inds_elem_cnt = reduce_across_ranks([output["total_pos_inds_elem_cnt"] for output in outputs])

        print(
            "{:<8} {:<16} {:<16.8f} {:<16.8f} {:<16.8f} {:<16.8f} {:<16.8f} {:<8}".format(
                iter,
                elapsed_time_str,
                loss_rpn_box_reg,
                loss_objectness,
                loss_box_reg,
                loss_classifier,
                loss_mask,
                int(total_pos_inds_elem_cnt),
            )
        )

        df = pd.DataFrame(
            [
                {"iter": iter, "legend": "elapsed_time", "value": elapsed_time},
                {
                    "iter": iter,
                    "legend": "loss_rpn_box_reg",
                    "value": loss_rpn_box_reg,
                },
                {
                    "iter": iter,
                    "legend": "loss_objectness",
                    "value": loss_objectness,
                },
                {"iter": iter, "legend": "loss_box_reg", "value": loss_box_reg},
                {
                    "iter": iter,
                    "legend": "loss_classifier",
                    "value": loss_classifier,
                },
                {"iter": iter, "legend": "loss_mask", "value": loss_mask},
                {
                    "iter": iter,
                    "legend": "total_pos_inds_elem_cnt",
                    "value": total_pos_inds_elem_cnt,
                },
            ]
        )
        metrics = pd.concat([metrics, df], axis=0)

    return metrics


def run():
    fake_image_list = []
    if hasattr(terminal_args, "fake_image_path"):
        file_list = os.listdir(terminal_args.fake_image_path)
        fake_image_list = [
            np.load(os.path.join(terminal_args.fake_image_path, f))
            for f in file_list
        ]

    # Get mrcn train function
    config = merge_args_to_train_config(terminal_args)
    train_func = init_train_func(config, len(fake_image_list) > 0)

    # model init
    check_point = flow.train.CheckPoint()
    check_point.init()
    # check_point.load(terminal_args.model_load_dir)
    if terminal_args.save_model_every_n_batch > 0:
        save_model(check_point, 0)

    start_time = time.time()
    elapsed_times = []
    metrics = pd.DataFrame(
        {"iter": 0, "legend": "cfg", "value": str(cfg)}, index=[0]
    )
    print_metric_title()
    for i in range(terminal_args.iter_num):
        if i < len(fake_image_list):
            outputs = train_func(fake_image_list[i]).get()
        else:
            outputs = train_func().get()

        now_time = time.time()
        elapsed_time = now_time - start_time
        elapsed_times.append(elapsed_time)
        start_time = now_time

        save_blob_watched(i)

        if terminal_args.save_model_every_n_batch > 0:
            if (
                (i + 1) % terminal_args.save_model_every_n_batch == 0
                or i + 1 == terminal_args.iter_num
            ):
                save_model(check_point, i + 1)

        metrics = update_metrics(metrics, i, elapsed_time, outputs)
        if terminal_args.save_loss_npy_every_n_batch > 0:
            if (
                (i + 1) % terminal_args.save_loss_npy_every_n_batch == 0
                or i + 1 == terminal_args.iter_num
            ):
                npy_file_name = "loss-{}-batch_size-{}-gpu-{}-image_dir-{}-{}.csv".format(
                    i,
                    terminal_args.batch_size,
                    terminal_args.gpu_num_per_node,
                    terminal_args.image_dir,
                    str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S")),
                )
                metrics.to_csv(npy_file_name, index=False)
                print("saved: {}".format(npy_file_name))

    print("median of elapsed time per batch:", statistics.median(elapsed_times))


if __name__ == "__main__":
    run()
