"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.nn.graph import GraphModule


rank = flow.env.get_rank()


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
            data_part_num=40,
            part_name_suffix_length=5,
            random_shuffle=False,
            shuffle_after_epoch=False,
            placement=placement,
            sbp=sbp,
            random_seed=0,
        )
        self.record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )

        color_space = "RGB"
        height = 22
        width = 22

        self.record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )

        self.resize = flow.nn.image.Resize(target_size=[height, width])

        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.batch_size

    def forward(self):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        image_raw_buffer = self.record_image_decoder(train_record)
        image = self.resize(image_raw_buffer)[0]
        image = flow.flatten(image.to(flow.float32), start_dim=1)

        return image, label


def _train_with_graph(iter_num=3):
    B = [flow.sbp.broadcast]
    P0 = flow.placement("cuda", ranks=[0])
    P1 = flow.placement("cuda", ranks=[1])
    P2 = flow.placement("cuda", ranks=[2])
    P3 = flow.placement("cuda", ranks=[3])

    train_data_loader = OFRecordDataLoader(
        ofrecord_root="/dataset/ImageNet/ofrecord",
        mode="train",
        dataset_size=400,
        batch_size=4,
        placement=P0,
        sbp=B,
    )

    def _get_ppm_and_opt():
        class StageModule(flow.nn.Module):
            def __init__(self, *linear_args):
                super().__init__()
                self.linear = flow.nn.Linear(*linear_args)
                flow.nn.init.constant_(self.linear.weight, 0.00023)

            def forward(self, input):
                out = self.linear(input)
                return out

        class PipelineModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                # Initlize module and move each module to the right placement of its pipeline stage.
                self.stage_0_m = StageModule(1452, 8, False).to_global(
                    placement=P0, sbp=B
                )
                self.stage_1_m = StageModule(8, 8, False).to_global(placement=P1, sbp=B)
                self.stage_2_m = StageModule(8, 8, False).to_global(placement=P2, sbp=B)
                self.stage_3_m = StageModule(8, 1, False).to_global(placement=P3, sbp=B)

            def forward(self, image):
                out = self.stage_0_m(image)
                # Move tensor between different pipeline stages.
                out = out.to_global(placement=P1, sbp=B)
                out = self.stage_1_m(out)
                out = out.to_global(placement=P2, sbp=B)
                out = self.stage_2_m(out)
                out = out.to_global(placement=P3, sbp=B)
                out = self.stage_3_m(out)
                return out

        pp_m = PipelineModule()
        sgd = flow.optim.SGD(pp_m.parameters(), lr=0.0001)
        return pp_m, sgd

    pp_m, sgd = _get_ppm_and_opt()

    class PipelineGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.train_data_loader = train_data_loader
            self.pp_m = pp_m
            # Set different module's stage id to hint the graph preparing right num of buffers in pipeline.
            self.pp_m.stage_0_m.to(GraphModule).set_stage(0)
            self.pp_m.stage_1_m.to(GraphModule).set_stage(1)
            self.pp_m.stage_2_m.to(GraphModule).set_stage(2)
            self.pp_m.stage_3_m.to(GraphModule).set_stage(3)
            self.mseloss = flow.nn.MSELoss("sum")
            self.add_optimizer(sgd)
            # Let graph to do gradient accumulatioin, pipline execution depends on gradient accumulatioin.
            self.config.set_gradient_accumulation_steps(4)

        def build(self):
            image, label = self.train_data_loader()

            # Dataloader's outputs are on host memory, so move it to device 0.
            image = image.to_global(placement=P0, sbp=B)
            pp_m.train()
            out = self.pp_m(image)

            # Dataloader's outputs are on host memory, so move it to device 3.
            label = label.to_global(placement=P3, sbp=B)
            loss = self.mseloss(out, label.to(dtype=flow.float32))
            loss.backward()

            # Returning image and label is just for re-using data in eager test
            image = image.to_global(placement=P3, sbp=B)
            return loss, image, label

    pp_g = PipelineGraph()

    def one_iter(iter_idx):
        loss, image, label = pp_g()
        if rank == 3:
            # loss on other rank are 0-Size tensor
            loss = loss.to_local()
            loss_np = loss.numpy()
            print("loss numpy \n", loss)
            image = image.to_local().numpy()
            label = label.to_local().numpy()
            return loss, image, label

    check_list = []
    data_list = []
    for i in range(iter_num):
        out = one_iter(i)
        if rank == 3:
            check_list.append(out[0])
            data_list.append((out[1], out[2]))
    return check_list, data_list


def _train_with_module(iter_num=3, data=None):
    class DataModule(flow.nn.Module):
        def __init__(self, data):
            super().__init__()
            self.data_list = []
            self.idx = 0
            for pair in data:
                for i in range(4):
                    s = i * 4
                    e = s + 4
                    micro_batch_image = pair[0][s:e]
                    micro_batch_label = pair[1][s:e]
                    self.data_list.append(
                        (
                            flow.Tensor(micro_batch_image).to("cuda:3"),
                            flow.Tensor(micro_batch_label).to("cuda:3"),
                        )
                    )

        def forward(self):
            image = self.data_list[self.idx][0]
            label = self.data_list[self.idx][1]
            self.idx += 1
            return image, label

    class TrainModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(1452, 8, False)
            flow.nn.init.constant_(self.linear.weight, 0.00023)
            self.linear.to("cuda:3")
            self.linear1 = flow.nn.Linear(8, 8, False)
            flow.nn.init.constant_(self.linear1.weight, 0.00023)
            self.linear1.to("cuda:3")
            self.linear2 = flow.nn.Linear(8, 8, False)
            flow.nn.init.constant_(self.linear2.weight, 0.00023)
            self.linear2.to("cuda:3")
            self.linear3 = flow.nn.Linear(8, 1, False)
            flow.nn.init.constant_(self.linear3.weight, 0.00023)
            self.linear3.to("cuda:3")
            self.mseloss = flow.nn.MSELoss("sum")

        def forward(self, image, label):
            out = self.linear(image)
            out = self.linear1(out)
            out = self.linear2(out)
            out = self.linear3(out)
            loss = self.mseloss(out, label)
            return loss

    if rank == 3:
        data_m = DataModule(data)
        train_m = TrainModule()
        sgd = flow.optim.SGD(train_m.parameters(), lr=0.0001)

        def one_iter(iter_idx):
            if rank == 3:
                image, label = data_m()
                loss = train_m(image, label)

                loss_np = loss.numpy()
                print("eager loss numpy \n", loss_np)

                loss = loss * 0.25
                loss.backward()
                if iter_idx % 4 == 3:
                    print(f"iter index: {iter_idx}")
                    # eager gradient accumulatioin
                    sgd.step()
                    sgd.zero_grad()
                return loss_np

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter(i))
        return check_list


def _test_graph_pipeline(test_case):
    iter_num = 3
    graph_check_list, data = _train_with_graph(iter_num)
    module_check_list = _train_with_module(iter_num * 4, data)

    if rank == 3:
        for i in range(iter_num * 4):
            # check equal on loss
            test_case.assertTrue(
                np.array_equal(module_check_list[i], graph_check_list[i // 4][i % 4])
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n4d()
class TestGraphPipeline(oneflow.unittest.TestCase):
    def test_graph_pipeline(test_case):
        _test_graph_pipeline(test_case)


if __name__ == "__main__":
    unittest.main()
