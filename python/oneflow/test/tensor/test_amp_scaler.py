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
from itertools import chain
import os
import pickle
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest
import random


class TestGradScaling(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_grad_scaling_device_as_key(test_case):
        # Ensure that different instances of "device" objects that point to the same device
        # are treated as identical keys by dicts.  GradScaler relies on this behavior, and may
        # error otherwise in a way that's difficult to detect (a silent performance hit).
        d = {}
        t = flow.empty((1,), device="cuda:0")
        dev0a = flow.device("cuda:0")
        dev0b = flow.device("cuda:0")
        dev1a = flow.device("cuda:1")
        dev1b = flow.device("cuda:1")

        test_case.assertTrue(hash(dev0a) == hash(dev0b))
        test_case.assertTrue(hash(dev1a) == hash(dev1b))

        d[dev0a] = "0a"
        d[dev0b] = "0b"
        test_case.assertTrue(len(d) == 1)
        test_case.assertTrue(d[dev0a] == "0b")
        d[t.device] = "t"
        test_case.assertTrue(len(d) == 1)
        test_case.assertTrue(d[dev0a] == "t")

        d[dev1a] = "1a"
        d[dev1b] = "1b"
        test_case.assertTrue(len(d) == 2)
        test_case.assertTrue(d[dev1a] == "1b")

    @flow.unittest.skip_unless_1n1d()
    def test_grad_scaling_scale(test_case):
        scaler = flow.cuda.amp.GradScaler(
            init_scale=2.0, fused=random.choice([True, False])
        )
        t0 = flow.full((1,), 4.0, dtype=flow.float32, device="cuda:0")
        t1 = flow.full((1,), 4.0, dtype=flow.float32, device="cuda:1")
        # Create some nested iterables of tensors on different devices.
        outputs = (
            t1.clone(),
            (t0.clone(), t1.clone()),
            [t0.clone(), (t1.clone(), t0.clone())],
        )
        outputs = scaler.scale(outputs)
        test_case.assertTrue(
            outputs[0] == 8.0
            and outputs[1][0] == 8.0
            and outputs[1][1] == 8.0
            and outputs[2][0] == 8.0
            and outputs[2][1][0] == 8.0
            and outputs[2][1][1] == 8.0
        )
        test_case.assertTrue(scaler._scale.device == t1.device)

    @flow.unittest.skip_unless_1n1d()
    def test_grad_scaling_state_dict(test_case):
        for lazy_init_scale in True, False:
            s0 = flow.cuda.amp.GradScaler(
                init_scale=3.0,
                growth_factor=4.0,
                backoff_factor=0.5,
                growth_interval=2,
                fused=random.choice([True, False]),
            )
            s1 = flow.cuda.amp.GradScaler(
                init_scale=6.0,
                growth_factor=7.0,
                backoff_factor=0.8,
                growth_interval=1,
                fused=random.choice([True, False]),
            )

            # sets a random value for load_state_dict to overwrite
            s1._init_growth_tracker = 7

            if lazy_init_scale:
                # Dummy scale() call to ensure the scale tensor is lazily initialized.
                s1.scale(flow.full((1,), 4.0, dtype=flow.float32, device="cuda:0"))
                test_case.assertTrue(isinstance(s1._scale, flow.Tensor))

            s1.load_state_dict(s0.state_dict())

            test_case.assertEqual(s1.get_scale(), 3.0)
            test_case.assertEqual(s1.get_growth_factor(), 4.0)
            test_case.assertEqual(s1.get_backoff_factor(), 0.5)
            test_case.assertEqual(s1.get_growth_interval(), 2)
            test_case.assertEqual(s1._init_growth_tracker, 0)

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
            scaler = flow.cuda.amp.GradScaler(
                init_scale=128.0,
                growth_factor=2.0,
                enabled=enabled,
                growth_interval=1,
                fused=random.choice([True, False]),
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
                test_case.assertTrue(
                    np.allclose(c.numpy(), s.numpy(), atol=atol, rtol=1e-05)
                )

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
                        model[1].weight.grad.data.fill_(float("inf"))
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
        try_pickle = True
        test_case._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)

    def test_grad_scaling_update_scale(test_case, device="cuda", dtype=flow.float):
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        scale = flow.full((1,), 4.0, dtype=dtype, device=device)
        growth_tracker = flow.full((1,), 0.0, dtype=flow.int32, device=device)
        found_inf = flow.full((1,), 0.0, dtype=flow.float, device="cuda:0")

        # Simulates 2 consecutive unskipped iterations
        flow._C.amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        test_case.assertEqual(growth_tracker, 1)
        test_case.assertEqual(scale, 4.0)
        flow._C.amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        test_case.assertEqual(growth_tracker, 0)
        test_case.assertEqual(scale, 8.0)

        # Simulates a skipped iteration
        found_inf.fill_(1.0)
        flow._C.amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        test_case.assertEqual(growth_tracker, 0)
        test_case.assertEqual(scale, 2.0)

    def test_grad_scaling_clipping(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    flow.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm * scaler.get_scale()
                    )
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    flow.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-5)

    def test_grad_scaling_clipping_separate_unscale(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.unscale_(optimizer)
                    flow.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm, error_if_nonfinite=False
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    flow.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_penalty(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = flow.autograd.grad(
                        outputs=scaler.scale(loss),
                        inputs=list(model.parameters()),
                        create_graph=True,
                    )[0]
                    inv_scale = 1.0 / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = flow.autograd.grad(
                        outputs=loss, inputs=list(model.parameters()), create_graph=True
                    )[0]

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_accumulation(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            iters_to_accumulate = 2
            for i, (input, target) in enumerate(data):
                output = model(input)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate
                if try_scaling_api:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if try_scaling_api:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

        self._run_scaling_case(run, unskipped=2, skipped=0)

    def test_grad_scaling_multiple(test_case):
        # Tests gradient scaling with 2 models and 2 optimizers that both receive gradients from 2 losses.
        # Some of the logic here cannot reuse the generic helper functions created for the 1-optimizer cases.
        for enabled in True, False:
            (
                mod_control0,
                mod_scaling0,
                opt_control0,
                opt_scaling0,
                data,
                loss_fn,
                skip_iter,
            ) = test_case._create_scaling_case()
            (
                mod_control1,
                mod_scaling1,
                opt_control1,
                opt_scaling1,
            ) = test_case._create_scaling_models_optimizers()

            scaler = flow.cuda.amp.GradScaler(
                init_scale=128.0,
                growth_factor=2.0,
                enabled=enabled,
                growth_interval=1,
                fused=random.choice([True, False]),
            )

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float("inf"))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            test_case.assertTrue(
                scaler.get_scale()
                == (
                    128.0
                    * scaler.get_growth_factor() ** 3
                    * scaler.get_backoff_factor() ** 1
                )
                if enabled
                else 1.0
            )

            for c, s in zip(
                chain(mod_control0.parameters(), mod_control1.parameters()),
                chain(mod_scaling0.parameters(), mod_scaling1.parameters()),
            ):
                test_case.assertTrue(
                    np.allclose(c.numpy(), s.numpy(), rtol=1e-5, atol=1e-7)
                )

    @flow.unittest.skip_unless_1n2d()
    def test_grad_scaling_multigpu(test_case):
        # Same as above, but runs some of the models on device 1.
        # GradScaler should transparently handle losses and gradients on multiple devices.
        # This test could be combined with the test above, but I think it makes sense to treat
        # multi-GPU operations separately.
        dev0 = flow.device("cuda:0")
        dev1 = flow.device("cuda:1")

        for enabled in True, False:
            (
                mod_control0,
                mod_scaling0,
                opt_control0,
                opt_scaling0,
                data,
                loss_fn,
                skip_iter,
            ) = test_case._create_scaling_case()
            (
                mod_control1,
                mod_scaling1,
                opt_control1,
                opt_scaling1,
            ) = test_case._create_scaling_models_optimizers(device=dev1)

            scaler = flow.cuda.amp.GradScaler(
                init_scale=128.0,
                growth_factor=2.0,
                enabled=enabled,
                growth_interval=1,
                fused=random.choice([True, False]),
            )

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input.to(dev1))
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1.to(dev0), target)
                    loss1 = loss_fn(
                        0.6 * output0.to(dev1) - 0.4 * output1, target.to(dev1)
                    )

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float("inf"))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)

                        # Make sure the found_infs were collected properly across optimizers and devices.
                        if scaler.is_enabled():
                            test_case.assertTrue(
                                len(scaler._found_inf_per_device(optimizer0)) == 1
                            )
                            test_case.assertTrue(
                                len(scaler._found_inf_per_device(optimizer1)) == 1
                            )
                            test_case.assertTrue(
                                scaler._found_inf_per_device(optimizer0)[dev0].item()
                                == 0.0
                            )
                            test_case.assertTrue(
                                scaler._found_inf_per_device(optimizer1)[dev1].item()
                                == float(i == skip_iter)
                            )

                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            test_case.assertTrue(
                scaler.get_scale()
                == (
                    128.0
                    * scaler.get_growth_factor() ** 3
                    * scaler.get_backoff_factor() ** 1
                )
                if enabled
                else 1.0
            )

            # Copy mod_control1 and mod_scaling1 back the device 0 for comparison
            mod_control1.to(dev0)
            mod_scaling1.to(dev0)

            for c, s in zip(
                chain(mod_control0.parameters(), mod_control1.parameters()),
                chain(mod_scaling0.parameters(), mod_scaling1.parameters()),
            ):
                test_case.assertTrue(
                    np.allclose(c.numpy(), s.numpy(), rtol=1e-5, atol=1e-7)
                )



if __name__ == "__main__":
    unittest.main()
