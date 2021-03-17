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
import unittest

import oneflow as flow


class TestEagerModel(flow.unittest.TestCase):
    def test_model_validate(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x):
                return x + self.w

        class EagerModel(flow.Model):
            def __init__(self):
                super().__init__()
                self.m = CustomModule(2)

            def forward(self, x):
                return self.m(x)

            def validation_step(self, batch):
                return self(batch)

        class ValData(flow.model.DataModule):
            def __init__(self):
                super().__init__()

            def forward(self, *args):
                return 3

        class OutputMonitor(flow.model.Callback):
            def on_validation_step_end(self, step_idx, outputs):
                test_case.assertEqual(outputs, 5)
                fmt_str = "{:>12}  {:>12}  {:>12.6f}"
                print(fmt_str.format(step_idx, "validation output:", outputs))

        val_exe_config = flow.ExecutionConfig()
        val_config = flow.model.ValidationConfig()
        val_config.config_execution(val_exe_config)
        val_config.config_data(ValData())
        val_config.config_step_interval(1)

        output_monitor_cb = OutputMonitor()

        eager_md = EagerModel()

        eager_md.fit(
            validation_config=val_config, callbacks=output_monitor_cb, max_steps=10
        )


if __name__ == "__main__":
    unittest.main()
