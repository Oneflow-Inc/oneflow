import os
import sys

# For debug
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8003"
os.environ["WORLD_SIZE"] = "4"
os.environ["RANK"] = str(sys.argv[1])
os.environ["LOCAL_RANK"] = str(sys.argv[1])

import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest

class OFRecordDataLoader(flow.nn.Module):
    def __init__(
        self,
        ofrecord_root: str = "./ofrecord",
        mode: str = "train",  # "val"
        dataset_size: int = 9469,
        batch_size: int = 1,
        placement=None,
        sbp=None,
    ):
        super().__init__()
        channel_last = False
        output_layout = "NHWC" if channel_last else "NCHW"
        self.train_record_reader = flow.nn.OFRecordReader(
            ofrecord_root + "/" + mode,
            batch_size=batch_size,
            data_part_num=4,
            part_name_suffix_length=5,
            random_shuffle=False if mode == "train" else False,
            shuffle_after_epoch=False if mode == "train" else False,
            placement=placement,
            sbp=sbp
        )
        self.record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )

        color_space = "RGB"
        height = 224
        width = 224

        self.record_image_decoder = flow.nn.OFRecordImageDecoder("encoded", color_space=color_space)

        self.resize = (
            flow.nn.image.Resize(target_size=[height, width])
            if mode == "train"
            else flow.nn.image.Resize(
                resize_side="shorter", keep_aspect_ratio=True, target_size=256
            )
        )

        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        self.crop_mirror_norm = (
            flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )
            if mode == "train"
            else flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                crop_h=height,
                crop_w=width,
                crop_pos_y=0.5,
                crop_pos_x=0.5,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )
        )

        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.batch_size

    def forward(self):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        image_raw_buffer = self.record_image_decoder(train_record)
        image = self.resize(image_raw_buffer)[0]
        rng = None
        image = self.crop_mirror_norm(image, rng)

        return image, label

D = "cuda"
B = [flow.sbp.broadcast]
P = flow.placement("cuda", {0: [0, 1, 2, 3]})
P0 = flow.placement("cuda", {0: [0]})
P1 = flow.placement("cuda", {0: [1]})
P2 = flow.placement("cuda", {0: [2]})
P3 = flow.placement("cuda", {0: [3]})

def _get_ppm_and_opt():
    train_data_loader = OFRecordDataLoader(
        ofrecord_root="/dataset/ImageNet/ofrecord",
        mode="train",
        dataset_size=400,
        batch_size=4,
        placement=P0,
        sbp=B,
    )

    class Stage0Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.train_data_loader = train_data_loader
            self.linear = flow.nn.Linear(224, 8, False)
            self.linear.to_consistent(placement=P0, sbp=B)
            flow.nn.init.constant_(self.linear.weight, 0.023)

        def forward(self):
            image, label = self.train_data_loader()
            image = image.to(D)
            label = label.to(D)
            out0 = self.linear(image)
            return out0, label

    class Stage1Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(8, 8, False)
            self.linear.to_consistent(placement=P1, sbp=B)
            flow.nn.init.constant_(self.linear.weight, 0.023)
        
        def forward(self, input):
            out0 = input.to_consistent(placement=P1, sbp=input.sbp)
            out1 = self.linear(out0)
            return out1

    class Stage2Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(8, 8, False)
            self.linear.to_consistent(placement=P2, sbp=B)
            flow.nn.init.constant_(self.linear.weight, 0.023)
        
        def forward(self, input):
            out0 = input.to_consistent(placement=P2, sbp=input.sbp)
            out1 = self.linear(out0)
            return out1

    class Stage3Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(8, 2, False)
            self.linear.to_consistent(placement=P3, sbp=B)
            flow.nn.init.constant_(self.linear.weight, 0.023)
        
        def forward(self, out0, label):
            out0 = out0.to_consistent(placement=P3, sbp=out0.sbp)
            label = label.to_consistent(placement=P3, sbp=out0.sbp)
            out1 = self.linear(out0)
            loss = label.to(flow.float32).sum() - out1.sum() 
            return loss

    class PipelineModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.stage_0_m = Stage0Module()
            self.stage_1_m = Stage1Module()
            self.stage_2_m = Stage2Module()
            self.stage_3_m = Stage3Module()

        def forward(self):
            out0, label = self.stage_0_m()
            out1 = self.stage_1_m(out0)
            out2 = self.stage_2_m(out1)
            out3 = self.stage_3_m(out2, label)
            return out3

    pp_m = PipelineModule()
    of_sgd = flow.optim.SGD(pp_m.parameters(), lr=0.001, momentum=0.0)
    return pp_m, of_sgd

def _test_graph_pipeline(test_case):
    rank = flow.env.get_rank()

    def train_with_graph(iter_num=3):
        pp_m, of_sgd = _get_ppm_and_opt()

        class PipelineGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.pp_m = pp_m
                self.pp_m.stage_0_m.config.stage_id = 0
                self.pp_m.stage_1_m.config.stage_id = 1
                self.pp_m.stage_2_m.config.stage_id = 2
                self.pp_m.stage_3_m.config.stage_id = 3
                self.config.set_gradient_accumulation_steps(4)
                self.add_optimizer(of_sgd)

            def build(self):
                pp_m.train()
                out = self.pp_m()
                out.backward()
                return out

        pp_g = PipelineGraph()

        def one_iter(iter_idx):
            of_graph_out = pp_g()
            if rank == 3:
                of_graph_out = of_graph_out.to_local()
                of_graph_out_np = of_graph_out.numpy()
                print("out numpy ", of_graph_out_np)
                return of_graph_out_np

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter(i))
        return check_list

    def train_with_module(iter_num=3):
        train_data_loader = OFRecordDataLoader(
            ofrecord_root="/dataset/ImageNet/ofrecord",
            mode="train",
            dataset_size=400,
            batch_size=4,
            #placement=P0,
            #sbp=B,
        )
        class TrainModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.train_data_loader = train_data_loader
                self.linear = flow.nn.Linear(224, 8, False)
                flow.nn.init.constant_(self.linear.weight, 0.023)
                self.linear1 = flow.nn.Linear(8, 8, False)
                flow.nn.init.constant_(self.linear1.weight, 0.023)
                self.linear2 = flow.nn.Linear(8, 8, False)
                flow.nn.init.constant_(self.linear2.weight, 0.023)
                self.linear3 = flow.nn.Linear(8, 2, False)
                flow.nn.init.constant_(self.linear3.weight, 0.023)

            def forward(self):
                image, label = self.train_data_loader()
                out0 = self.linear(image)
                out1 = self.linear1(out0)
                out2 = self.linear2(out1)
                out3 = self.linear3(out2)
                loss = label.to(flow.float32).sum() - out3.sum() 
                return loss
        
        t_m = TrainModule()
        of_sgd = flow.optim.SGD(t_m.parameters(), lr=0.001, momentum=0.0)

        def one_iter(iter_idx):
            loss = t_m()
            out_np = loss.numpy()
            print("eager out numpy ", out_np)
            #loss = loss * 0.25
            loss.backward()
            if iter_idx % 4 == 3:
                print(f"iter index: {iter_idx}")
                of_sgd.step()
                of_sgd.zero_grad()
            return out_np 

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter(i))
        return check_list

    iter_num = 3

    if (rank == 3):
        module_check_list = train_with_module(iter_num * 4)

    graph_check_list = train_with_graph(iter_num)

    #if (rank == 1):
    #    for i in range(iter_num):
    #        # check equal on loss
    #        test_case.assertTrue(
    #            np.array_equal(module_check_list[i][0], graph_check_list[i][0])
    #        )
    #        # check equal on weight
    #        test_case.assertTrue(
    #            np.array_equal(module_check_list[i][1], graph_check_list[i][1])
    #        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphPipeline(oneflow.unittest.TestCase):
    def test_graph_pipeline(test_case):
        _test_graph_pipeline(test_case)


if __name__ == "__main__":
    sys.argv.pop()
    unittest.main()
