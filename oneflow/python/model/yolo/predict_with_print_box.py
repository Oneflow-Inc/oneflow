import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net import YoloPredictNet

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-t", "--train_dir", type=str, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default="../yolov/of_model/yolov3_model_python/", required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default="", required=False
)
parser.add_argument("-class_num", "--class_num", type=int, default=85742, required=False)
parser.add_argument("-batch_size", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-total_batch_num", "--total_batch_num", type=int, default=310, required=False)
parser.add_argument("-num_of_batches_in_snapshot", "--num_of_batches_in_snapshot", type=int, required=False)
parser.add_argument("-lr", "--base_lr", type=float, default=0, required=False)
parser.add_argument("-weight_l2", "--weight_l2", type=float, default=0, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)

args = parser.parse_args()

def yolo_decode(name):
    with flow.fixed_placement("cpu", "0:0"):
        return flow.user_op_builder(name).Op("yolo_decoder").Output("out").Output("origin_image_info").Build().RemoteBlobList()

flow.config.load_library("predict_decoder_op.so")
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.use_tensorrt(True)
func_config.tensorrt.use_fp16()

label_2_name=[]

with open('coco.names','r') as f:
  label_2_name=f.readlines()

#nms=False
nms=True
print("nms:", nms)
def print_detect_box(positions, probs):
    if nms==True:
        for i in range(1, 81):
            for j in range(positions.shape[1]):
                if positions[i][j][1]!=0 and positions[i][j][2]!=0 and probs[i][j]!=0:
                    print(label_2_name[i-1], " ", probs[i][j]*100,"%", "  ", positions[i][j][0], " ", positions[i][j][1], " ", positions[i][j][2], " ", positions[i][j][3])
    else:
        for j in range(positions.shape[1]):
            for i in range(1, 81):
                if positions[0][j][1]!=0 and positions[0][j][2]!=0 and probs[0][j][i]!=0:
                    print(label_2_name[i-1], " ", probs[0][j][i]*100,"%", "  ",positions[0][j][0], " ", positions[0][j][1], " ", positions[0][j][2], " ", positions[0][j][3])


@flow.function(func_config)
def yolo_user_op_eval_job():
    images, origin_image_info = yolo_decode("yolo")
    images = flow.identity(images, name="yolo-layer1-start")
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    yolo_pos_result = flow.identity(yolo_pos_result, name="yolo_pos_result_end")
    yolo_prob_result = flow.identity(yolo_prob_result, name="yolo_prob_result_end")
    return yolo_pos_result, yolo_prob_result

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
            yolo_pos, yolo_prob = ret
            #print(yolo_pos.shape, yolo_pos)
            #print(yolo_prob.shape, yolo_prob)
            print_detect_box(yolo_pos, yolo_prob)
            #np.save("tmp/pos-step"+str(step), yolo_pos.ndarray())
            #np.save("tmp/prob-step"+str(step), yolo_prob.ndarray())
            global cur_time
            if step==0:
                print("start_time:", time.time())
            elif step==args.total_batch_num-1:
                print("end time:", time.time())
            print(time.time()-cur_time)

            cur_time = time.time()

        if step % args.loss_print_steps == 0:
            return callback
        else:
            return nop



    for step in range(args.total_batch_num):
        yolo_user_op_eval_job().async_get(create_callback(step))
