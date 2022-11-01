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
import pickle
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest


class TestGradScaling(flow.unittest.TestCase):
    # @flow.unittest.skip_unless_1n1d()
    # def test_grad_scaling_device_as_key(test_case):
    #     # Ensure that different instances of "device" objects that point to the same device
    #     # are treated as identical keys by dicts.  GradScaler relies on this behavior, and may
    #     # error otherwise in a way that's difficult to detect (a silent performance hit).
    #     d = {}
    #     t = flow.empty((1,), device="cuda:0")
    #     dev0a = flow.device("cuda:0")
    #     dev0b = flow.device("cuda:0")
    #     dev1a = flow.device("cuda:1")
    #     dev1b = flow.device("cuda:1")

    #     test_case.assertTrue(hash(dev0a) == hash(dev0b))
    #     test_case.assertTrue(hash(dev1a) == hash(dev1b))

    #     d[dev0a] = "0a"
    #     d[dev0b] = "0b"
    #     test_case.assertTrue(len(d) == 1)
    #     test_case.assertTrue(d[dev0a] == "0b")
    #     d[t.device] = "t"
    #     test_case.assertTrue(len(d) == 1)
    #     test_case.assertTrue(d[dev0a] == "t")

    #     d[dev1a] = "1a"
    #     d[dev1b] = "1b"
    #     test_case.assertTrue(len(d) == 2)
    #     test_case.assertTrue(d[dev1a] == "1b")

    # @flow.unittest.skip_unless_1n1d()
    # def test_grad_scaling_scale(test_case):
    #     scaler = flow.amp.GradScaler(init_scale=2.)
    #     t0 = flow.full((1,), 4.0, dtype=flow.float32, device="cuda:0")
    #     t1 = flow.full((1,), 4.0, dtype=flow.float32, device="cuda:1")
    #     # Create some nested iterables of tensors on different devices.
    #     outputs = (t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), (t1.clone(), t0.clone())])
    #     outputs = scaler.scale(outputs)
    #     test_case.assertTrue(outputs[0] == 8.0 and outputs[1][0] == 8.0 and outputs[1][1] == 8.0 and
    #                     outputs[2][0] == 8.0 and outputs[2][1][0] == 8.0 and outputs[2][1][1] == 8.0)
    #     test_case.assertTrue(scaler._scale.device == t1.device)

    # @flow.unittest.skip_unless_1n1d()
    # def test_grad_scaling_state_dict(test_case):
    #     for lazy_init_scale in True, False:
    #         s0 = flow.amp.GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
    #         s1 = flow.amp.GradScaler(init_scale=6., growth_factor=7., backoff_factor=.8, growth_interval=1)

    #         # sets a random value for load_state_dict to overwrite
    #         s1._init_growth_tracker = 7

    #         if lazy_init_scale:
    #             # Dummy scale() call to ensure the scale tensor is lazily initialized.
    #             s1.scale(flow.full((1,), 4.0, dtype=flow.float32, device="cuda:0"))
    #             test_case.assertTrue(isinstance(s1._scale, flow.Tensor))

    #         s1.load_state_dict(s0.state_dict())

    #         test_case.assertEqual(s1.get_scale(), 3.)
    #         test_case.assertEqual(s1.get_growth_factor(), 4.)
    #         test_case.assertEqual(s1.get_backoff_factor(), .5)
    #         test_case.assertEqual(s1.get_growth_interval(), 2)
    #         test_case.assertEqual(s1._init_growth_tracker, 0)

    def _create_scaling_models_optimizers(self, device="cuda"):
        # Create a module+optimizer that will use scaling, and a control module+optimizer
        # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
        mod_control = flow.nn.Sequential(flow.nn.Linear(8, 8), flow.nn.Linear(8, 8)).to(
            device=device
        )
        mod_scaling = flow.nn.Sequential(flow.nn.Linear(8, 8), flow.nn.Linear(8, 8)).to(
            device=device
        )
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.data.copy_(c.data)

        opt_control = flow.optim.SGD(mod_control.parameters(), lr=1.0)
        opt_scaling = flow.optim.SGD(mod_scaling.parameters(), lr=1.0)

        return mod_control, mod_scaling, opt_control, opt_scaling

    def _create_scaling_case(self, device="cuda", dtype=flow.float):
        data = [
            (
                flow.randn((8, 8), dtype=dtype, device=device),
                flow.randn((8, 8), dtype=dtype, device=device),
            ),
            (
                flow.randn((8, 8), dtype=dtype, device=device),
                flow.randn((8, 8), dtype=dtype, device=device),
            ),
            (
                flow.randn((8, 8), dtype=dtype, device=device),
                flow.randn((8, 8), dtype=dtype, device=device),
            ),
            (
                flow.randn((8, 8), dtype=dtype, device=device),
                flow.randn((8, 8), dtype=dtype, device=device),
            ),
        ]

        loss_fn = flow.nn.MSELoss().cuda()

        skip_iter = 2

        return self._create_scaling_models_optimizers(device=device) + (
            data,
            loss_fn,
            skip_iter,
        )

    # _run_scaling_case generalizes some single-optimizer test logic to avoid too much copy-pasting below.
    def _run_scaling_case(test_case, run, unskipped, skipped, atol=1e-7):
        # Ensure scaling can be disabled without changing user control flow.
        for enabled in True, False:
            (
                mod_control,
                mod_scaling,
                opt_control,
                opt_scaling,
                data,
                loss_fn,
                skip_iter,
            ) = test_case._create_scaling_case()

            # For functionality, test with a modest initial scale, and an unrealistically-large growth factor
            # so any potential errors with the growth factor handling will be magnified.
            scaler = flow.amp.GradScaler(
                init_scale=128.0, growth_factor=2.0, enabled=enabled, growth_interval=1
            )

            _ = run(data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # Allows run() to optionally return a different scaler instance.
            scaler = ret if ret else scaler

            # If scaling was enabled, the scale factor should have been multiplied by the growth factor
            # len(data) - skipped times and the backoff factor "skipped" times.
            if enabled:
                net_growth = (
                    scaler.get_growth_factor() ** unskipped if unskipped > 0 else 1.0
                )
                net_backoff = (
                    scaler.get_backoff_factor() ** skipped if skipped > 0 else 1.0
                )
                test_case.assertTrue(
                    scaler.get_scale() == (128.0 * net_growth * net_backoff)
                )
            else:
                test_case.assertTrue(scaler.get_scale() == 1.0)

            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                test_case.assertEqual(np.allclose(c.numpy(), s.numpy(), atol=atol, rtol=1e-05), True)

    # Compares no scaling + no autocasting against scaling + autocasting.
    @flow.unittest.skip_unless_1n1d()
    def test_grad_scaling_autocast(test_case):
        try_pickle = False

        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                device = "cpu"
                if try_scaling_api:
                    device = "cuda"
                with flow.autocast(device):
                    output = model(input)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
        test_case._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)
        # this will be picked up by try_pickle within run():
        # try_pickle = True
        # test_case._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
