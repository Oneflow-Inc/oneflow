import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net import YoloTrainNet

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-t", "--train_dir", type=str, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default="of_model/yolov3_model_python/", required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default="", required=False
)
parser.add_argument("-class_num", "--class_num", type=int, default=85742, required=False)
parser.add_argument("-batch_size", "--batch_size", type=int, default=4, required=False)
parser.add_argument("-data_part_num", "--data_part_num", type=int, required=False)
parser.add_argument("-part_name_suffix_length", "--part_name_suffix_length", type=int, default=-1, required=False)
parser.add_argument("-total_batch_num", "--total_batch_num", type=int, default=100, required=False)
parser.add_argument("-num_of_batches_in_snapshot", "--num_of_batches_in_snapshot", type=int, required=False)
parser.add_argument("-lr", "--base_lr", type=float, default=0.1, required=False)
parser.add_argument("-weight_l2", "--weight_l2", type=float, default=0, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-gt_max", "--gt_max_len", type=int, default=90, required=False)
parser.add_argument("-s", "--shuffle", type=int, default=1, required=False)
args = parser.parse_args()

def yolo_decode(name):
    with flow.fixed_placement("cpu", "0:0"):
        return flow.user_op_builder(name).Op("yolo_train_decoder")\
            .Output("data").Output("ground_truth").Output("gt_valid_num")\
            .SetAttr("batch_size", args.batch_size, "AttrTypeInt32").Build().RemoteBlobList()

def ParameterUpdateStrategy():
    return {
        'learning_rate_decay': {
             'piecewise_scaling_conf' : {
             'boundaries': [400000, 450000],
             'scales': [1.0, 0.1, 0.01]
             }
        },
        'momentum_conf': {
            'beta': 0.9
        }
    }


flow.config.load_library("train_decoder_op.so")
#flow.config.enable_debug_mode(True)
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(args.base_lr)
func_config.train.weight_l2(args.weight_l2)
func_config.train.model_update_conf(ParameterUpdateStrategy())


@flow.function(func_config)
def yolo_train_job():
    (images_, ground_truth_, gt_valid_num_) = yolo_decode("my_yolo")
    gt_boxes = flow.slice(ground_truth_, [None, 0, 0], [None, -1, 4], name = 'gt_box')
    gt_labels = flow.cast(flow.slice(ground_truth_, [None, 0, 4], [None, -1, 1], name = 'gt_label'), dtype=flow.int32)
    print(images_.shape, gt_boxes.shape,gt_labels.shape, gt_valid_num_.shape)
    yolo0_loss, yolo1_loss, yolo2_loss = YoloTrainNet(images_, gt_boxes, gt_labels, gt_valid_num_, True)
    flow.losses.add_loss(yolo0_loss)
    flow.losses.add_loss(yolo1_loss)
    flow.losses.add_loss(yolo2_loss)
    return yolo0_loss, yolo1_loss, yolo2_loss


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.ctrl_port(9789)

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}   {:>12.10f} {:>12.10f} {:>12.3f}"
    print("{:>12}   {:>12}  {:>12}  {:>12}".format("iter",  "reg loss value", "cls loss value", "time"))
    global cur_time
    cur_time = time.time()
    
    def create_callback(step):
        def nop(ret):
            pass
        def callback(ret):
            yolo0_loss, yolo1_loss, yolo2_loss = ret
            print(yolo0_loss.mean(), yolo1_loss.mean(), yolo2_loss.mean())
            global cur_time
            print(time.time()-cur_time)
            cur_time = time.time()

        if step % args.loss_print_steps == 0:
            return callback
        else:
            return nop


    for step in range(args.total_batch_num):
        yolo_train_job().async_get(create_callback(step))
