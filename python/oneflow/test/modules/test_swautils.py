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
r"""
This test module references to pytorch.
https://github.com/pytorch/pytorch/blob/master/test/test_optim.py.
"""
import math
import unittest
import itertools
import numpy as np

import oneflow as flow
import oneflow.optim as optim
import oneflow.nn.functional as F
from oneflow.nn import Parameter
from oneflow.optim import SGD, Optimizer
from oneflow.nn.optimizer.lr_scheduler import LRScheduler
from oneflow.nn.optimizer.multiplicative_lr import MultiplicativeLR
from oneflow.nn.optimizer.swa_utils import AveragedModel, SWALR, update_bn
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestLRScheduler(flow.unittest.TestCase):
    # This class mainly used to test MultiplicativeLR and SWALR
    def setUp(self):
        super(TestLRScheduler, self).setUp()
        self.net = SchedulerTestNet()
        self.opt = SGD(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

    def test_multiplicative_lr(self):
        # test Multiplicative lr
        epochs = 10
        self.opt.param_groups[0]["lr"] = 0.05
        self.opt.param_groups[1]["lr"] = 0.4
        targets = [
            [0.05 * (0.9 ** x) for x in range(epochs)],
            [0.4 * (0.8 ** x) for x in range(epochs)],
        ]
        scheduler = MultiplicativeLR(
            self.opt, lr_lambda=[lambda x1: 0.9, lambda x2: 0.8]
        )
        self._test(scheduler, targets, epochs)

    def _test(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertTrue(
                    np.allclose(
                        target[epoch], param_group["lr"], atol=1e-6, rtol=1e-5,
                    ),
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                )
            [scheduler.step() for scheduler in schedulers]

    def test_swa_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: SWALR(self.opt, anneal_epochs=3, swa_lr=0.5),
            lambda: SWALR(
                self.opt, anneal_epochs=10, anneal_strategy="linear", swa_lr=5.0
            ),
        )

    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        scheduler = constr()
        for _ in range(epochs):
            scheduler.optimizer.step()
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key != "optimizer":
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])
        self.assertEqual(scheduler.get_last_lr(), scheduler_copy.get_last_lr())

    def test_swalr_no_anneal(self):
        epochs, swa_start, swa_lr = 10, 5, 0.01
        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets = [
            [lr] * (swa_start + 1) + [swa_lr] * (epochs - swa_start - 1)
            for lr in initial_lrs
        ]
        swa_scheduler = SWALR(self.opt, anneal_epochs=1, swa_lr=swa_lr)
        self._test_swalr(swa_scheduler, None, targets, swa_start, epochs)

    def test_swalr_cosine_anneal_after_multiplicative(self):
        # same swa_lr for different param_groups
        epochs, swa_start, swa_lr, anneal_epochs = 15, 5, 0.01, 5
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(self.opt, anneal_epochs=anneal_epochs, swa_lr=swa_lr)

        def anneal_coef(t):
            if t + 1 >= anneal_epochs:
                return 0.0
            return (1 + math.cos(math.pi * (t + 1) / anneal_epochs)) / 2

        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets_before_swa = [
            [lr * mult_factor ** i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        swa_epochs = epochs - swa_start - 1
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs in targets_before_swa
        ]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def _test_swalr(self, swa_scheduler, scheduler, targets, swa_start, epochs):
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertTrue(
                    np.allclose(
                        target[epoch], param_group["lr"], atol=1e-6, rtol=1e-5,
                    ),
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                )
            if epoch >= swa_start:
                self.opt.step()
                swa_scheduler.step()
            elif scheduler is not None:
                self.opt.step()
                scheduler.step()

    def test_swalr_hypers(self):
        # Test that SWALR raises errors for incorrect hyper-parameters
        with self.assertRaisesRegex(ValueError, "anneal_strategy must"):
            swa_scheduler = SWALR(self.opt, anneal_strategy="exponential", swa_lr=1.0)

        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=-1, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=1.7, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "swa_lr must"):
            swa_scheduler = SWALR(self.opt, swa_lr=[1.0, 0.1, 0.01])


@flow.unittest.skip_unless_1n1d()
class TestSWAUtils(flow.unittest.TestCase):
    # This class mainly used to test AveragedModel and update_bn
    def _test_averaged_model(self, net_device, swa_device):
        # test the average of AveragedModel
        dnn = flow.nn.Sequential(
            flow.nn.Conv2d(1, 5, kernel_size=3),
            flow.nn.ReLU(),
            flow.nn.MaxPool2d(kernel_size=2),
            flow.nn.BatchNorm2d(5, momentum=0.3),
            flow.nn.Conv2d(5, 2, kernel_size=3),
            flow.nn.ReLU(),
            flow.nn.Linear(5, 5),
            flow.nn.ReLU(),
            flow.nn.Linear(5, 10),
        ).to(net_device)

        averaged_dnn = AveragedModel(dnn, device=swa_device)
        averaged_params = [flow.zeros_like(param) for param in dnn.parameters()]
        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(flow.randn_like(p))
                p_avg += p.detach() / n_updates
            if i == 0:
                averaged_dnn.update_parameters(dnn)
            else:
                averaged_dnn.update_parameters(dnn)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertTrue(
                flow.allclose(p_avg.cpu(), p_swa.cpu(), atol=1e-5, rtol=1e-4)
            )
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_swa.device == swa_device)
            self.assertTrue(p.device == net_device)
        self.assertTrue(averaged_dnn.n_averaged.device == swa_device)

    def test_averaged_model_all_devices(self):
        cpu = flow.device("cpu")
        self._test_averaged_model(cpu, cpu)
        if flow.cuda.is_available():
            cuda = flow.device("cuda:0")
            self._test_averaged_model(cuda, cpu)
            self._test_averaged_model(cpu, cuda)
            self._test_averaged_model(cuda, cuda)

    def test_averaged_model_mixed_device(self):
        if not flow.cuda.is_available():
            return
        dnn = flow.nn.Sequential(
            flow.nn.Conv2d(1, 5, kernel_size=3), flow.nn.Linear(5, 10)
        )
        dnn[0].cuda()
        dnn[1].cpu()
        averaged_dnn = AveragedModel(dnn)
        averaged_params = [flow.zeros_like(param) for param in dnn.parameters()]
        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(flow.randn_like(p))
                p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertTrue(flow.allclose(p_avg, p_swa, atol=1e-5, rtol=1e-4))
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_avg.device == p_swa.device)

    def test_averaged_model_state_dict(self):
        dnn = flow.nn.Sequential(
            flow.nn.Conv2d(1, 5, kernel_size=3), flow.nn.Linear(5, 10)
        )
        averaged_dnn = AveragedModel(dnn)
        averaged_dnn2 = AveragedModel(dnn)
        n_updates = 10
        for i in range(n_updates):
            for p in dnn.parameters():
                p.detach().add_(flow.randn_like(p))
            averaged_dnn.update_parameters(dnn)
        averaged_dnn2.load_state_dict(averaged_dnn.state_dict())
        for p_swa, p_swa2 in zip(averaged_dnn.parameters(), averaged_dnn2.parameters()):
            self.assertTrue(flow.allclose(p_swa, p_swa2, atol=1e-5, rtol=1e-4))
        self.assertTrue(averaged_dnn.n_averaged == averaged_dnn2.n_averaged)

    def test_averaged_model_exponential(self):
        # Test AveragedModel with EMA as avg_fn
        dnn = flow.nn.Sequential(
            flow.nn.Conv2d(1, 5, kernel_size=3),
            flow.nn.BatchNorm2d(5, momentum=0.3),
            flow.nn.Linear(5, 10),
        )
        alpha = 0.9

        def avg_fn(p_avg, p, n_avg):
            return alpha * p_avg + (1 - alpha) * p

        averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn)
        averaged_params = [flow.zeros_like(param) for param in dnn.parameters()]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(flow.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * alpha + p * (1 - alpha)).clone()
                    )
            for b in dnn.buffers():
                if b.size() != flow.Size([]):
                    # oneflow don't support detach_
                    # b.detach_().add_(flow.randn_like(b))
                    b.detach().add_(flow.randn_like(b))

            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertTrue(flow.allclose(p_avg, p_swa, atol=1e-5, rtol=1e-4))
        for b_avg, b_swa in zip(dnn.buffers(), averaged_dnn.module.buffers()):
            self.assertTrue(flow.allclose(b_avg, b_swa, atol=1e-5, rtol=1e-4))

    def test_averaged_model_exponential_buffers(self):
        # Test AveragedModel with EMA as avg_fn and use_buffers as True.
        dnn = flow.nn.Sequential(
            flow.nn.Conv2d(1, 5, kernel_size=3),
            flow.nn.BatchNorm2d(5, momentum=0.3),
            flow.nn.Linear(5, 10),
        )
        alpha = 0.9

        def avg_fn(p_avg, p, n_avg):
            return alpha * p_avg + (1 - alpha) * p

        averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn, use_buffers=True)
        dnn_params = itertools.chain(dnn.parameters(), dnn.buffers())
        averaged_params = [
            flow.zeros_like(param)
            for param in dnn_params
            if param.size() != flow.Size([])
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(dnn_params, averaged_params):
                if p.size() == flow.Size.Size([]):
                    continue
                p.detach().add_(flow.Size.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * alpha + p * (1 - alpha)).clone()
                    )
            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        for p_avg, p_swa in zip(
            averaged_params,
            itertools.chain(
                averaged_dnn.module.parameters(), averaged_dnn.module.buffers()
            ),
        ):
            self.assertTrue(flow.allclose(p_avg, p_swa, atol=1e-5, rtol=1e-4))

    def _test_update_bn(self, dnn, dl_x, dl_xy, momentum, cuda):

        preactivation_sum = flow.zeros(dnn.n_features)
        preactivation_squared_sum = flow.zeros(dnn.n_features)
        if cuda:
            preactivation_sum = preactivation_sum.cuda()
            preactivation_squared_sum = preactivation_squared_sum.cuda()
        total_num = 0
        for x in dl_x:
            x = x[0]
            if cuda:
                x = x.cuda()

            dnn.forward(x)
            preactivations = dnn.compute_preactivation(x)
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            preactivations = preactivations.contiguous().view(-1, dnn.n_features)
            total_num += preactivations.shape[0]

            preactivation_sum += flow.sum(preactivations, dim=0)
            preactivation_squared_sum += flow.sum(preactivations ** 2, dim=0)

        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean ** 2

        update_bn(dl_xy, dnn, device=x.device)
        self.assertTrue(
            flow.allclose(preactivation_mean, dnn.bn.running_mean, atol=1e-6, rtol=1e-3)
        )
        self.assertTrue(
            flow.allclose(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=1e-1)
        )

        def _reset_bn(module):
            if issubclass(module.__class__, flow.nn.modules.batchnorm._BatchNorm):
                module.running_mean = flow.zeros_like(module.running_mean)
                module.running_var = flow.ones_like(module.running_var)

        # reset batch norm and run update_bn again
        dnn.apply(_reset_bn)
        update_bn(dl_xy, dnn, device=x.device)
        self.assertTrue(
            flow.allclose(preactivation_mean, dnn.bn.running_mean, atol=1e-6, rtol=1e-3)
        )
        self.assertTrue(
            flow.allclose(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=1e-1)
        )
        # using the dl_x loader instead of dl_xy
        dnn.apply(_reset_bn)
        update_bn(dl_x, dnn, device=x.device)
        self.assertTrue(
            flow.allclose(preactivation_mean, dnn.bn.running_mean, atol=1e-6, rtol=1e-3)
        )
        self.assertTrue(
            flow.allclose(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=1e-1)
        )

    def test_update_bn_dnn(self):
        # Test update_bn for a fully-connected network with BatchNorm1d
        objects, input_features = 100, 5
        x = flow.rand(objects, input_features)
        y = flow.rand(objects)
        ds_x = flow.utils.data.TensorDataset(x)
        ds_xy = flow.utils.data.TensorDataset(x, y)
        dl_x = flow.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = flow.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = SWATestDNN(input_features=input_features)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, 0.1, False)
        if flow.cuda.is_available():
            dnn = SWATestDNN(input_features=input_features)
            dnn.train()
            self._test_update_bn(dnn.cuda(), dl_x, dl_xy, 0.1, True)
        self.assertTrue(dnn.training)

    def test_update_bn_cnn(self):
        # Test update_bn for convolutional network and BatchNorm2d
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = flow.rand(objects, input_channels, height, width)
        y = flow.rand(objects)
        ds_x = flow.utils.data.TensorDataset(x)
        ds_xy = flow.utils.data.TensorDataset(x, y)
        dl_x = flow.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = flow.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = SWATestCNN(input_channels=input_channels)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, 0.3, False)
        if flow.cuda.is_available():
            dnn = SWATestCNN(input_channels=input_channels)
            dnn.train()
            self._test_update_bn(dnn.cuda(), dl_x, dl_xy, 0.3, True)
        self.assertTrue(dnn.training)

    def test_bn_update_eval_momentum(self):
        # check that update_bn preserves eval mode
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = flow.rand(objects, input_channels, height, width)
        ds_x = flow.utils.data.TensorDataset(x)
        dl_x = flow.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dnn = SWATestCNN(input_channels=input_channels)
        dnn.eval()
        update_bn(dl_x, dnn)
        self.assertFalse(dnn.training)

        # check that momentum is preserved
        self.assertEqual(dnn.bn.momentum, 0.3)


class SWATestDNN(flow.nn.Module):
    def __init__(self, input_features):
        super(SWATestDNN, self).__init__()
        self.n_features = 100
        self.fc1 = flow.nn.Linear(input_features, self.n_features)
        self.bn = flow.nn.BatchNorm1d(self.n_features)

    def compute_preactivation(self, x):
        return self.fc1(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return x


class SWATestCNN(flow.nn.Module):
    def __init__(self, input_channels):
        super(SWATestCNN, self).__init__()
        self.n_features = 10
        self.conv1 = flow.nn.Conv2d(
            input_channels, self.n_features, kernel_size=3, padding=1
        )
        self.bn = flow.nn.BatchNorm2d(self.n_features, momentum=0.3)

    def compute_preactivation(self, x):
        return self.conv1(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x


class SchedulerTestNet(flow.nn.Module):
    def __init__(self):
        super(SchedulerTestNet, self).__init__()
        self.conv1 = flow.nn.Conv2d(1, 1, 1)
        self.conv2 = flow.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


if __name__ == "__main__":
    unittest.main()
