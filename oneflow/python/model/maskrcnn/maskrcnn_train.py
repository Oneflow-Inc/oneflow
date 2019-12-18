# must import get cfg before importing oneflow
from config import get_default_cfgs, compare_config

import os
import numpy as np
import pandas as pd
import argparse
import time
import statistics
import oneflow as flow

from datetime import datetime
from backbone import Backbone
from rpn import RPNHead, RPNLoss, RPNProposal, gen_anchors
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
parser.add_argument("-fake", "--fake_image_path", type=str, required=False)
parser.add_argument(
    "-save",
    "--model_save_dir",
    type=str,
    default="./model_save-{}".format(
        str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    ),
    required=False,
)
parser.add_argument(
    "-pr",
    "--print_loss_each_rank",
    default=False,
    action="store_true",
    required=False,
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
    rpn_proposal = RPNProposal(cfg, True)
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

    anchors = gen_anchors(
        image,
        cfg.MODEL.RPN.ANCHOR_STRIDE,
        cfg.MODEL.RPN.ANCHOR_SIZES,
        cfg.MODEL.RPN.ASPECT_RATIOS,
    )

    # Backbone
    # CHECK_POINT: fpn features
    # with flow.watch_scope(
    #     blob_watcher=blob_watched
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


def merge_and_compare_config(args):
    config = get_default_cfgs()

    if hasattr(args, "config_file"):
        config.merge_from_file(args.config_file)
        print("merged config from {}".format(args.config_file))

    assert config.SOLVER.IMS_PER_BATCH % config.ENV.NUM_GPUS == 0
    config.ENV.IMS_PER_GPU = config.SOLVER.IMS_PER_BATCH / config.ENV.NUM_GPUS
    config.freeze()
    print("difference between default (upper) and given config (lower)")
    d1_diff, d2_diff = compare_config(get_default_cfgs(), config)
    print("")
    print("default:\n{}\n".format(str(d1_diff)))
    print("given:\n{}\n".format(str(d2_diff)))
    if args.verbose:
        print(config)

    return config


def set_train_config(cfg):
    flow.config.cudnn_buf_limit_mbyte(cfg.ENV.CUDNN_BUFFER_SIZE_LIMIT)
    flow.config.cudnn_conv_heuristic_search_algo(
        cfg.ENV.CUDNN_CONV_HEURISTIC_SEARCH_ALGO
    )
    flow.config.cudnn_conv_use_deterministic_algo_only(
        cfg.ENV.CUDNN_CONV_USE_DETERMINISTIC_ALGO_ONLY
    )
    assert cfg.MODEL.WEIGHT
    flow.config.default_initialize_with_snapshot_path(cfg.MODEL.WEIGHT)
    flow.config.train.primary_lr(cfg.SOLVER.BASE_LR)
    flow.config.train.secondary_lr(
        cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    )
    flow.config.train.weight_l2(cfg.SOLVER.WEIGHT_DECAY)
    flow.config.train.bias_l2(cfg.SOLVER.WEIGHT_DECAY_BIAS)

    if cfg.SOLVER.OPTIMIZER == "momentum":
        optimizer = dict(momentum_conf={"beta": cfg.SOLVER.MOMENTUM})
    elif cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = dict(naive_conf={})
    else:
        raise ValueError("optimizer must be 'momentum' or 'sgd'")

    if cfg.SOLVER.ENABLE_LR_DECAY:
        optimizer.update(
            dict(
                learning_rate_decay=dict(
                    piecewise_scaling_conf={
                        "boundaries": cfg.SOLVER.STEPS,
                        "scales": cfg.SOLVER.LR_DECAY_VALUES,
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
                            "warmup_batches": cfg.SOLVER.WARMUP_ITERS,
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
    return flow.config.train.get_model_update_conf()


def save_model(check_point, i):
    if not os.path.exists(terminal_args.model_save_dir):
        os.makedirs(terminal_args.model_save_dir)
    model_dst = os.path.join(terminal_args.model_save_dir, "iter-" + str(i))
    print("saving models to {}".format(model_dst))
    check_point.save(model_dst)


def load_iter_num_from_file(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            import struct

            (iter_num,) = struct.unpack("q", f.read())
            return iter_num
    else:
        return None


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


def get_flow_dtype(dtype_str):
    if dtype_str == "float32":
        return flow.float32
    else:
        raise NotImplementedError

def make_lr(train_step_name, model_update_conf, primary_lr, secondary_lr=None):
    # usually, train_step_name is "System-Train-TrainStep-" + train job name
    assert model_update_conf.HasField("learning_rate_decay") or model_update_conf.HasField("warmup_conf"), "only support model update conf with warmup or lr decay for now"
    flow.config.train.train_step_lbn(train_step_name + "-Identity" + "/out")
    secondary_lr_lbn = "System-Train-SecondaryLearningRate-Scheduler/out"
    if secondary_lr is None:
        secondary_lr_lbn = "System-Train-PrimaryLearningRate-Scheduler/out"
    flow.config.train.lr_lbn("System-Train-PrimaryLearningRate-Scheduler/out", "System-Train-SecondaryLearningRate-Scheduler/out")
    # these two lines above must be called before creating any op
    with flow.device_prior_placement("cpu", "0:0"):
        train_step = flow.get_variable(
            name=train_step_name,
            shape=(1,),
            dtype=flow.int64,
            initializer=flow.constant_initializer(0),
            trainable=False
        )
        train_step_id = flow.identity(train_step, name=train_step_name + "-Identity")
        flow.assign(train_step, train_step_id + 1, name=train_step_name + "-Assign")
        
        primary_lr_blob = flow.schedule(train_step_id, model_update_conf, primary_lr, name="System-Train-PrimaryLearningRate-Scheduler")
        secondary_lr_blob = None
        if secondary_lr is None:
            secondary_lr_blob = primary_lr_blob
        else:
            secondary_lr_blob = flow.schedule(train_step_id, model_update_conf, secondary_lr, name="System-Train-SecondaryLearningRate-Scheduler")
        assert secondary_lr_blob is not None
        
        return {
            "train_step": train_step_id,
            "lr": primary_lr_blob,
            "lr2": secondary_lr_blob
        }
        
def init_train_func(config, input_fake_image):
    flow.env.ctrl_port(terminal_args.ctrl_port)
    flow.config.enable_inplace(config.ENV.ENABLE_INPLACE)
    flow.config.gpu_device_num(config.ENV.NUM_GPUS)
    flow.config.default_data_type(get_flow_dtype(config.DTYPE))

    if input_fake_image:

        @flow.function
        def train(
            image_blob=flow.input_blob_def(
                shape=(2, 800, 1344, 3), dtype=flow.float32, is_dynamic=True
            )
        ):
            set_train_config(config)
            return train_net(config, image_blob)

        return train

    else:

        @flow.function
        def train():
            model_update_conf = set_train_config(config)
            step_lr = make_lr("System-Train-TrainStep-train", model_update_conf, flow.config.train.get_primary_lr(), flow.config.train.get_secondary_lr())
            outputs = train_net(config)
            if isinstance(outputs, (list, tuple)):
                outputs[0].update(step_lr)
            else:
                outputs.update(step_lr)
            return outputs

        return train

def transpose_metrics(metrics):
    legends = metrics["legend"].unique()
    transposed = metrics.pivot_table(
        values="value", columns=["legend"], aggfunc="mean", dropna=False
    )
    assert metrics["iter"].unique().size == 1, "can only transpose metrics in one iter"
    transposed["iter"] = metrics["iter"].unique()
    return transposed

def print_metrics(m):
    to_print_with_order = [
        "iter",
        "rank",
        "loss_rpn_box_reg",
        "loss_objectness",
        "loss_box_reg",
        "loss_classifier",
        "loss_mask",
        "total_pos_inds_elem_cnt",
        "train_step",
        "lr",
        "lr2",
    ]
    to_print_with_order = [l for l in to_print_with_order if l in m]
    print(m[to_print_with_order].to_string(index=False))

def add_metrics(metrics_df, iter=None, **kwargs):
    assert iter is not None
    for k, v in kwargs.items():
        if k is "outputs":
            if isinstance(v, list):
                dfs = []
                for rank, v in enumerate(v, 0):
                    for legend, value in v.items():
                        dfs.append(pd.DataFrame({"iter": iter, "rank": rank, "legend": legend, "value": value.item()}, index=[0]))
            elif isinstance(v, dict):
                dfs = [pd.DataFrame(
                    {"iter": iter, "legend": legend, "value": value.item()}, index=[0]
                ) for legend, value in v.items()]
            else:
                raise ValueError("not supported")
            metrics_df = pd.concat([metrics_df] + dfs, axis=0, sort=False)
        elif k is not "outputs":
            metrics_df = pd.concat([metrics_df, pd.DataFrame(
                    {"iter": iter, "legend": k, "value": v}, index=[0]
                )], axis=0, sort=False)
        else:
            raise ValueError("not supported")
    return metrics_df

def run():
    use_fake_images = False
    if hasattr(terminal_args, "fake_image_path"):
        use_fake_images = True
        file_list = os.listdir(terminal_args.fake_image_path)
        fake_image_list = [
            np.load(os.path.join(terminal_args.fake_image_path, f))
            for f in file_list
        ]

    # Get mrcn train function
    config = merge_and_compare_config(terminal_args)
    train_func = init_train_func(config, use_fake_images)

    # model init
    check_point = flow.train.CheckPoint()
    check_point.init()
    # check_point.load(terminal_args.model_load_dir)
    if config.SOLVER.CHECKPOINT_PERIOD > 0:
        save_model(check_point, 0)

    start_time = time.time()
    elapsed_times = []

    iter_file = os.path.join(
        config.MODEL.WEIGHT,
        "System-Train-TrainStep-{}".format(train_func.__name__),
        "out",
    )
    loaded_iter = load_iter_num_from_file(iter_file)

    start_iter = 1
    if loaded_iter is None:
        print("{} not found, iter starts at 1".format(iter_file))
    else:
        print("{} found, last iter: {}".format(iter_file, loaded_iter))
        start_iter = loaded_iter + 1
    assert start_iter <= config.SOLVER.MAX_ITER, "{} vs {}".format(
        start_iter, config.SOLVER.MAX_ITER
    )

    metrics = pd.DataFrame(
        {"iter": start_iter, "legend": "cfg", "note": str(config)}, index=[0]
    )

    if use_fake_images:
        assert len(fake_image_list) >= config.SOLVER.MAX_ITER - start_iter + 1

    for i in range(start_iter, config.SOLVER.MAX_ITER + 1):
        if use_fake_images:
            outputs = train_func(fake_image_list[i - start_iter]).get()
        else:
            outputs = train_func().get()

        now_time = time.time()
        elapsed_time = now_time - start_time
        elapsed_time_str = "{:.6f}".format(elapsed_time)
        elapsed_times.append(elapsed_time)
        start_time = now_time

        save_blob_watched(i)

        if config.SOLVER.CHECKPOINT_PERIOD > 0:
            if (
                i % config.SOLVER.CHECKPOINT_PERIOD == 0
                or i == config.SOLVER.MAX_ITER
            ):
                save_model(check_point, i)

        metrics_df = pd.DataFrame()
        metrics_df = add_metrics(metrics_df, iter=i, elapsed_time=elapsed_time)
        metrics_df = add_metrics(metrics_df, iter=i, outputs=outputs)
        rank_size = metrics_df["rank"].dropna().unique().size if "rank" in metrics_df else 0
        if terminal_args.print_loss_each_rank and rank_size > 1:
            for rank_i in range(rank_size):
                tansposed = transpose_metrics(metrics_df[metrics_df["rank"] == rank_i])
                tansposed["rank"] = rank_i
                print_metrics(tansposed)
        else:
            tansposed = transpose_metrics(metrics_df)
            print_metrics(tansposed)
        metrics = pd.concat([metrics, metrics_df], axis=0, sort=False)
        if config.SOLVER.METRICS_PERIOD > 0:
            if (
                i % config.SOLVER.METRICS_PERIOD == 0
                or i == config.SOLVER.MAX_ITER
            ):
                npy_file_name = "loss-{}-batch_size-{}-gpu-{}-image_dir-{}-{}.csv".format(
                    i,
                    config.SOLVER.IMS_PER_BATCH,
                    config.ENV.NUM_GPUS,
                    config.DATASETS.IMAGE_DIR_TRAIN,
                    str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S")),
                )
                metrics.to_csv(npy_file_name, index=False)
                print("saved: {}".format(npy_file_name))

    print("median of elapsed time per batch:", statistics.median(elapsed_times))


if __name__ == "__main__":
    run()
