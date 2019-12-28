import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from darknet53_nopack import YoloTrainNet

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-t", "--train_dir", type=str, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default="", required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default="", required=False
)
parser.add_argument("-class_num", "--class_num", type=int, default=85742, required=True)
parser.add_argument("-batch_size", "--batch_size", type=int, required=True)
parser.add_argument("-data_part_num", "--data_part_num", type=int, required=True)
parser.add_argument("-part_name_suffix_length", "--part_name_suffix_length", type=int, default=-1, required=False)
parser.add_argument("-total_batch_num", "--total_batch_num", type=int, required=True)
parser.add_argument("-num_of_batches_in_snapshot", "--num_of_batches_in_snapshot", type=int, required=True)
parser.add_argument("-lr", "--base_lr", type=float, default=0, required=True)
parser.add_argument("-weight_l2", "--weight_l2", type=float, default=0, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-gt_max", "--gt_max_len", type=int, default=256, required=True)
parser.add_argument("-s", "--shuffle", type=int, default=1, required=True)
parser.add_argument("-r", "--raw_data", type=int, default=0, required=False)
args = parser.parse_args()

def _data_load_layer(data_dir, gt_max_len):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(608, 608, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec(image_preprocessors=[flow.data.ImagePreprocessor("bgr2rgb"), flow.data.ImageResizePreprocessor(width=608, height=608)]),
    )

    gt_box_blob_conf = flow.data.BlobConf(
        "gt_bbox", shape=(gt_max_len, 4), dtype=flow.float, codec=flow.data.RawCodec()
    )

    gt_label_blob_conf = flow.data.BlobConf(
        "gt_bbox_label", shape=(gt_max_len, 1), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    gt_valid_blob_conf = flow.data.BlobConf(
        "gt_valid_num", shape=(1,), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (gt_box_blob_conf, gt_label_blob_conf, gt_valid_blob_conf, image_blob_conf),
        batch_size=args.batch_size, data_part_num=args.data_part_num, part_name_suffix_length=args.part_name_suffix_length, name="decode"
    )

input_blob_def_dict = {
    "images" : flow.FixedTensorDef((args.batch_size, 608, 608, 3), dtype=flow.float),
    "gt_boxes" : flow.FixedTensorDef((args.batch_size, args.gt_max_len, 4), dtype=flow.float),
    "gt_labels" : flow.FixedTensorDef((args.batch_size, args.gt_max_len, 1),dtype=flow.int32),
    "gt_valid_num": flow.FixedTensorDef((args.batch_size, 1),dtype=flow.int32),
}

flow.config.load_library("new_yolo_train_decoder_multithread_op.so")
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(0.1)
func_config.train.model_update_conf(dict(naive_conf={}))

func_config1 = flow.FunctionConfig()
func_config1.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config1.default_data_type(flow.float)
num_piece_in_batch=2


@flow.function(func_config1)
def load_data_job():
    (gt_boxes, gt_labels, gt_valid_num, images) = _data_load_layer(args.train_dir, gt_max_len=args.gt_max_len)
    return (gt_boxes, gt_labels, gt_valid_num, images)

def preprocess_data(images, gt_boxes, gt_labels, gt_valid_num):
    images = images
    gt_boxes = gt_boxes
    gt_labels = gt_labels
    gt_valid_num = gt_valid_num
    images = np.ascontiguousarray(np.load("in_data1.npy").transpose(0, 2, 3, 1), dtype=np.float32)
    gt_boxes = np.load("gt_box1.npy").astype(np.float32)
    gt_labels = np.ascontiguousarray(np.load("gt_label1.npy").reshape(1, 90, 1), dtype=np.int32)
    gt_valid_num = np.load("gt_valid_num1.npy").astype(np.int32)
    return images, gt_boxes, gt_labels, gt_valid_num

@flow.function(func_config)
def yolo_train_job(images=input_blob_def_dict["images"], gt_boxes=input_blob_def_dict["gt_boxes"], gt_labels=input_blob_def_dict["gt_labels"], gt_valid_num=input_blob_def_dict["gt_valid_num"]):
    print(images.shape, gt_boxes.shape, gt_labels.shape, gt_valid_num.shape)
    yolo0_loss, yolo1_loss, yolo2_loss = YoloTrainNet(images, gt_boxes, gt_labels, gt_valid_num, True)
    flow.losses.add_loss(yolo0_loss)
    flow.losses.add_loss(yolo1_loss)
    flow.losses.add_loss(yolo2_loss)
    return yolo0_loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}   {:>12.10f} {:>12.10f} {:>12.3f}"
    print("{:>12}   {:>12}  {:>12}  {:>12}".format("iter",  "reg loss value", "cls loss value", "time"))
    global cur_time
    cur_time = time.time()


    for step in range(args.total_batch_num):
        (gt_boxes, gt_labels, gt_valid_num, images) = load_data_job().get()
        #images, gt_boxes, gt_labels, gt_valid_num = preprocess_data(images, gt_boxes, gt_labels, gt_valid_num)
        yolo0_loss = yolo_train_job(images.ndarray(), gt_boxes.ndarray(), gt_labels.ndarray(), gt_valid_num.ndarray()).get()
        print(yolo0_loss.mean())
        #save_blob_watched(step)
