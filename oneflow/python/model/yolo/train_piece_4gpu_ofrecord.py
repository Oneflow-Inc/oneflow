import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net_repeated import YoloTrainNet

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
parser.add_argument("-r", "--raw_data", type=int, default=0, required=False)
args = parser.parse_args()

#input_blob_def_dict = {
#    "images" : flow.input_blob_def((args.batch_size, 3, 608, 608), dtype=flow.float),
#    "gt_boxes" : flow.input_blob_def((args.batch_size, args.gt_max_len, 4), dtype=flow.float),
#    "gt_labels" : flow.input_blob_def((args.batch_size, args.gt_max_len, 1),dtype=flow.int32),
#    "gt_valid_num": flow.input_blob_def((args.batch_size, 1),dtype=flow.int32),
#}
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
flow.config.enable_debug_mode(True)
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(args.base_lr)
func_config.train.weight_l2(args.weight_l2)
#func_config.train.model_update_conf(dict(naive_conf={}))
func_config.train.model_update_conf(ParameterUpdateStrategy())
func_config1 = flow.FunctionConfig()
func_config1.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config1.default_data_type(flow.float)

#def yolo_decode(name):
#    with flow.fixed_placement("cpu", "0:0"):
#        return flow.user_op_builder(name).Op("yolo_train_decoder")\
#            .Output("data").Output("ground_truth").Output("gt_valid_num")\
#            .SetAttr("batch_size", args.batch_size, "AttrTypeInt32").Build().RemoteBlobList()
#        #return flow.user_op_builder(name).Op("yolo_train_decoder").Build().RemoteBlobList()
        

#@flow.function
#def YoloJob():
#    (images, ground_truth) = yolo_decode("my_yolo")
#    return images, ground_truth
#
#def preprocess_data(images, ground_truth):
#    print("images.shape", images.shape)
#    print("ground_truth.shape", ground_truth.shape)
#    #images = np.ascontiguousarray(images.transpose(0, 2, 3, 1), dtype=np.float32)
#    images = images
#    gt_boxes = np.ascontiguousarray(ground_truth[:, :, 0:4], dtype=np.float32)
#    gt_labels = np.ascontiguousarray(ground_truth[:, :, 4].reshape(ground_truth.shape[0], ground_truth.shape[1], 1), dtype=np.int32)
#    gt_valid_num = np.ones((gt_labels.shape[0], 1), dtype=np.int32)
#    for i in range(ground_truth.shape[0]):
#        for j in range(ground_truth.shape[1]):
#            if gt_labels[i][j] == 0 and gt_boxes[i][j][2] == 0 and gt_boxes[i][j][3] == 0:
#                gt_valid_num[i] = j
#                break
#
#    return images, gt_boxes, gt_labels, gt_valid_num
#
#@flow.function
#def yolo_train_job(images=input_blob_def_dict["images"], gt_boxes=input_blob_def_dict["gt_boxes"], gt_labels=input_blob_def_dict["gt_labels"], gt_valid_num=input_blob_def_dict["gt_valid_num"]):
#    flow.config.train.primary_lr(args.base_lr)
#    flow.config.train.weight_l2(args.weight_l2)
#    flow.config.enable_inplace(False)
#    flow.config.default_data_type(flow.float)
#    #flow.config.train.model_update_conf(ParameterUpdateStrategy())
#    flow.config.train.model_update_conf(dict(naive_conf={}))
#
#    yolo0_loss, yolo1_loss, yolo2_loss = YoloTrainNet(images, gt_boxes, gt_labels, gt_valid_num, True)
#    flow.losses.add_loss(yolo0_loss)
#    flow.losses.add_loss(yolo1_loss)
#    flow.losses.add_loss(yolo2_loss)
#    return yolo0_loss, yolo1_loss, yolo2_loss

def _data_load_layer_raw(data_dir, gt_max_len):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(608, 608, 3),
        dtype=flow.float,
        codec=flow.data.RawCodec(),
        )

    gt_box_blob_conf = flow.data.BlobConf(
        "gt_bbox", shape=(gt_max_len, 4), dtype=flow.float, codec=flow.data.RawCodec()
    )

    gt_label_blob_conf = flow.data.BlobConf(
        "gt_bbox_label", shape=(gt_max_len, 2), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (gt_box_blob_conf, gt_label_blob_conf, gt_valid_blob_conf, image_blob_conf),
        batch_size=args.batch_size, data_part_num=args.data_part_num, name="decode"
    )

def _data_load_layer(data_dir, gt_max_len):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(3, 608, 608),
        dtype=flow.float,
        codec=flow.data.RawCodec(),
        )

    gt_box_blob_conf = flow.data.BlobConf(
        "gt_bbox", shape=(gt_max_len, 4), dtype=flow.float, codec=flow.data.RawCodec()
    )

    gt_label_blob_conf = flow.data.BlobConf(
        "gt_bbox_label", shape=(gt_max_len, 1), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (gt_box_blob_conf, gt_label_blob_conf, image_blob_conf),
        batch_size=args.batch_size, data_part_num=args.data_part_num, part_name_suffix_length=args.part_name_suffix_length, name="decode"
    )

input_blob_def_dict = {
    "images" : flow.FixedTensorDef((args.batch_size, 3, 608, 608), dtype=flow.float),
    "gt_boxes" : flow.FixedTensorDef((args.batch_size, args.gt_max_len, 4), dtype=flow.float),
    "gt_labels" : flow.FixedTensorDef((args.batch_size, args.gt_max_len, 1),dtype=flow.int32),
    "gt_valid_num": flow.FixedTensorDef((args.batch_size, 1),dtype=flow.int32),
}

@flow.function(func_config1)
def load_data_job():
    (gt_boxes, gt_labels, images) = _data_load_layer(args.train_dir, gt_max_len=args.gt_max_len)
    return (gt_boxes, gt_labels, images)

def preprocess_data(images, gt_boxes, gt_labels):
    images = images
    gt_boxes = gt_boxes
    gt_labels = gt_labels
    gt_valid_num = np.ones((gt_boxes.shape[0], 1), dtype=np.int32)
    for i in range(gt_boxes.shape[0]):
        for j in range(gt_boxes.shape[1]):
            if gt_labels[i][j] == 0 and gt_boxes[i][j][2] == 0 and gt_boxes[i][j][3] == 0:
                gt_valid_num[i] = j
                break
    
    return images, gt_boxes, gt_labels, gt_valid_num



num_piece_in_batch=16#16

@flow.function(func_config)
def yolo_train_job(images=input_blob_def_dict["images"], gt_boxes=input_blob_def_dict["gt_boxes"], gt_labels=input_blob_def_dict["gt_labels"], gt_valid_num=input_blob_def_dict["gt_valid_num"]):
    with flow.fixed_placement("cpu", "0:0-3"):
        images_ = flow.unpack(flow.identity(images), num_piece_in_batch)
        gt_boxes_ = flow.unpack(flow.identity(gt_boxes), num_piece_in_batch)
        gt_labels_ = flow.unpack(flow.identity(gt_labels), num_piece_in_batch)
        gt_valid_num_ = flow.unpack(flow.identity(gt_valid_num), num_piece_in_batch)
    print(images_.shape, gt_boxes_.shape,gt_labels_.shape, gt_valid_num_.shape)
    yolo0_loss, yolo1_loss, yolo2_loss = YoloTrainNet(images_, gt_boxes_, gt_labels_, gt_valid_num_, True)
    flow.losses.add_loss(yolo0_loss)
    flow.losses.add_loss(yolo1_loss)
    flow.losses.add_loss(yolo2_loss)
    yolo0_loss = flow.pack(yolo0_loss, num_piece_in_batch)
    yolo1_loss = flow.pack(yolo1_loss, num_piece_in_batch)
    yolo2_loss = flow.pack(yolo2_loss, num_piece_in_batch)
    return yolo0_loss, yolo1_loss, yolo2_loss


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.ctrl_port(9789)

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12} {:>12.10f} {:>12.10f} {:>12.10f} {:>12.3f}"
    print("{:>12}  {:>12}  {:>12}  {:>12} {:>12}".format("iter", "yolo1 loss", "yolo2 loss", "yolo3 loss", "time"))
    global cur_time
    cur_time = time.time()
    
    def create_callback(step):
        def nop(ret):
            pass
        def callback(ret):
            yolo0_loss, yolo1_loss, yolo2_loss = ret
            global cur_time
            print(fmt_str.format(step, abs(yolo0_loss.mean()), abs(yolo1_loss.mean()), abs(yolo2_loss.mean()), time.time()-cur_time))
            cur_time = time.time()

        if step % args.loss_print_steps == 0:
            return callback
        else:
            return nop


    for step in range(args.total_batch_num):
        images, gt_boxes, gt_labels = YoloJob().get()
        images, gt_boxes, gt_labels, gt_valid_num = preprocess_data(images, ground_truth)
        yolo_train_job(images, gt_boxes, gt_labels, gt_valid_num).async_get(create_callback(step))
        #yolo_train_job().async_get(create_callback(step))
