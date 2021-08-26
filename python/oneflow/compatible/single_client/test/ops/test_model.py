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

import tempfile
import unittest

import numpy as np

import oneflow.compatible.single_client.unittest
from oneflow.compatible.single_client import experimental as flow
from oneflow.compatible.single_client.nn.parameter import Parameter


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestEagerModel(flow.unittest.TestCase):
    def test_model(test_case):
        model_dir = tempfile.TemporaryDirectory()
        model_dir_path = model_dir.name
        para = np.random.randn(2, 3)
        in_data = np.full((2, 3), 1)

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(flow.Tensor(para))

            def forward(self, x):
                return x + self.w

        class EagerModel(flow.Model):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()

            def forward(self, x):
                return self.m(x)

            def training_step(self, batch, **kwargs):
                return (flow.sum(self(batch)), list(self.m.parameters())[0])

            def configure_optimizers(self):
                sgd = flow.optim.SGD(
                    [
                        {
                            "params": list(self.m.parameters()),
                            "lr": 1.0,
                            "momentum": 0.0,
                            "scale": 1.0,
                        }
                    ]
                )
                return sgd

            def validation_step(self, batch):
                return self(batch)

        class TrainData(flow.model.DataModule):
            def __init__(self):
                super().__init__()

            def forward(self, step_idx=0, optimizer_idx=0):
                return flow.ones((2, 3))

        class ValData(flow.model.DataModule):
            def __init__(self):
                super().__init__()

            def forward(self, step_idx=0, optimizer_idx=0):
                return flow.ones((2, 3))

        class OutputMonitor(flow.model.Callback):
            def on_training_step_end(self, step_idx, outputs, optimizer_idx):
                nonlocal para
                loss = outputs[0].numpy()
                test_case.assertTrue(
                    np.allclose(
                        loss, np.sum(in_data + para, dtype=np.float), 0.0001, 0.0001
                    )
                )
                para -= 1
                test_case.assertTrue(
                    np.allclose(outputs[1].numpy(), para, 0.0001, 0.0001)
                )
                fmt_str = "{:>12}  {:>12}  {:>12.6f}"
                print(fmt_str.format(step_idx, "train loss:", loss.mean()))

            def on_validation_step_end(self, step_idx, outputs):
                nonlocal para
                test_case.assertTrue(
                    np.allclose(outputs.numpy(), in_data + para, 0.0001, 0.0001)
                )
                fmt_str = "{:>12}  {:>12}  {:>12.6f}"
                print(
                    fmt_str.format(
                        step_idx, "validation output:", outputs.numpy().mean()
                    )
                )

        train_config = flow.model.TrainingConfig()
        train_config.config_data(TrainData())
        val_config = flow.model.ValidationConfig()
        val_config.config_data(ValData())
        val_config.config_step_interval(1)
        ck_config = flow.model.CheckpointConfig()
        ck_config.config_save(dirpath=model_dir_path, step_interval=5)
        output_monitor_cb = OutputMonitor()
        eager_md = EagerModel()
        eager_md.fit(
            training_config=train_config,
            validation_config=val_config,
            checkpoint_config=ck_config,
            callbacks=output_monitor_cb,
            max_steps=10,
        )
        step_9_para = list(eager_md.parameters())[0][0].numpy()
        eager_md_load = EagerModel()
        eager_md_load.load_state_dict(flow.load(model_dir_path + "-4"))
        loaded_step_4_para = list(eager_md_load.parameters())[0][0].numpy()
        test_case.assertTrue(
            np.allclose(step_9_para, loaded_step_4_para - 5, 0.0001, 0.0001)
        )
        model_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
