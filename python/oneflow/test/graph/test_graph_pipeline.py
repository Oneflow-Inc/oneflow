import os
import sys

# For debug
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8003"
os.environ["WORLD_SIZE"] = "2"
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
            data_part_num=2,
            part_name_suffix_length=5,
            random_shuffle=True if mode == "train" else False,
            shuffle_after_epoch=True if mode == "train" else False,
            placement=placement,
            sbp=sbp
        )
        self.record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )

        color_space = "RGB"
        height = 224
        width = 224

        self.record_image_decoder = (
            flow.nn.OFRecordImageDecoderRandomCrop("encoded", color_space=color_space)
            if mode == "train"
            else flow.nn.OFRecordImageDecoder("encoded", color_space=color_space)
        )

        self.resize = (
            flow.nn.image.Resize(target_size=[height, width])
            if mode == "train"
            else flow.nn.image.Resize(
                resize_side="shorter", keep_aspect_ratio=True, target_size=256
            )
        )

        self.flip = flow.nn.CoinFlip(batch_size=batch_size, placement=placement, sbp=sbp) if mode == "train" else None

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
        rng = self.flip() if self.flip != None else None
        image = self.crop_mirror_norm(image, rng)

        return image, label


def _test_train_graph(test_case, device):
    rank = flow.env.get_rank()
    def train_with_module(iter_num=3):
        class LocalModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = flow.nn.Linear(3, 8, False)
                self.linear1 = flow.nn.Linear(8, 7, False)
                flow.nn.init.ones_(self.linear0.weight)
                flow.nn.init.constant_(self.linear1.weight, 2.3)

            def forward(self, x):
                out0 = self.linear0(x)
                out1 = self.linear1(out0)
                return out1

        local_m = LocalModule()
        local_m = local_m.to(device)

        of_sgd = flow.optim.SGD(local_m.parameters(), lr=0.001, momentum=0.9)

        x = flow.Tensor(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            device=device,
            requires_grad=False,
        )

        def one_iter():
            of_out = local_m(x)
            of_out = of_out.sum()

            of_out.backward()
            of_sgd.step()
            of_sgd.zero_grad()
            
            print("rank: ", rank, " eager out:", of_out.numpy())
            return of_out.numpy(), local_m.linear1.weight.numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    def train_with_graph(iter_num=3):
        D = "cuda"
        B = [flow.sbp.broadcast]
        P = flow.placement("cuda", {0: [0, 1]})
        P0 = flow.placement("cuda", {0: [0]})
        P1 = flow.placement("cuda", {0: [1]})

        train_data_loader = OFRecordDataLoader(
            ofrecord_root="/dataset/ImageNet/ofrecord",
            mode="train",
            dataset_size=400,
            batch_size=4,
            placement=P0,
            sbp=B,
        )
        class Stage1Module(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(8, 7, False)
                self.relu1 = flow.nn.ReLU()
                self.linear1.to_consistent(placement=P1, sbp=B)
                flow.nn.init.constant_(self.linear1.weight, 2.3)
            
            def forward(self, out0):
                out0 = out0.to_consistent(placement=P1, sbp=out0.sbp)
                out1 = self.linear1(out0)
                #out1 = self.relu1(out1)
                return out1

        class PipelineModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.train_data_loader = train_data_loader
                self.linear0 = flow.nn.Linear(224, 8, False)
                self.relu0 = flow.nn.ReLU()
                self.linear0.to_consistent(placement=P0, sbp=B)
                flow.nn.init.ones_(self.linear0.weight)
                self.stage1_m = Stage1Module()

            def forward(self):
                image, label = self.train_data_loader()
                image = image.to(D)
                label = label.to(D)
                out0 = self.linear0(image)
                #out0 = self.relu0(out0)
                out1 = self.stage1_m(out0)
                return out1

        pp_m = PipelineModule()

        of_sgd = flow.optim.SGD(pp_m.parameters(), lr=0.001, momentum=0.1)

        class PipelineGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.pp_m = pp_m
                self.pp_m.train_data_loader.config.stage_id =0
                self.pp_m.linear0.config.stage_id = 0
                self.pp_m.relu0.config.stage_id = 0
                self.pp_m.stage1_m.config.stage_id = 1
                # TODO(): support gradient accumulation
                #self.config.set_gradient_accumulation_steps(2)
                self.add_optimizer(of_sgd)

            def build(self):
                out = self.pp_m()
                out = out.sum()
                # TODO(): support partial placement of scalar tensor numpy()
                #out = out.to_consistent(placement=P, sbp=B)
                out.backward()
                print("out meta:", out._meta_repr())
                return out

        pp_g = PipelineGraph()

        def one_iter():
            pp_m.train()
            of_graph_out = pp_g()
            print("out sbp: ", of_graph_out.sbp)
            of_graph_out = of_graph_out.to_local()
            of_graph_out_np = of_graph_out.numpy()
            print("rank: ", rank, " pipeline graph out: ", of_graph_out_np)
            print("loss local", of_graph_out)
            print("loss local meta ", of_graph_out._meta_repr())
            print("loss local numel ", of_graph_out.numel())
            print(f"loss local ndim {of_graph_out.ndim}  shape {of_graph_out.shape}")
            return of_graph_out_np, pp_m.stage1_m.linear1.weight.to_local().numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    iter_num = 2
    #if (rank == 1):
    #    module_check_list = train_with_module(iter_num)

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
    def test_train_graph_gpu(test_case):
        _test_train_graph(test_case, flow.device("cuda"))


if __name__ == "__main__":
    sys.argv.pop()
    unittest.main()
