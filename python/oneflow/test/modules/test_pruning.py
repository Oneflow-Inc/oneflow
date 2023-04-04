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
import sys
from collections import OrderedDict

import numpy as np
import tempfile
import pickle
from oneflow.test_utils.test_util import GenArgList

import oneflow.nn.utils.prune as prune
import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn
import unittest.mock as mock
from contextlib import contextmanager

from oneflow.test_utils.automated_test_util import *


class TestPrune(flow.unittest.TestCase):
    def test_validate_pruning_amount_init(self):
        r"""Test the first util function that validates the pruning
            amount requested by the user the moment the pruning method
            is initialized. This test checks that the expected errors are
            raised whenever the amount is invalid.
            The original function runs basic type checking + value range checks.
            It doesn't check the validity of the pruning amount with
            respect to the size of the tensor to prune. That's left to
            `_validate_pruning_amount`, tested below.
            """
        # neither float not int should raise TypeError
        with self.assertRaises(TypeError):
            prune._validate_pruning_amount_init(amount="I'm a string")

        # float not in [0, 1] should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=1.1)
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=20.0)

        # negative int should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=-10)

        # all these should pass without errors because they're valid amounts
        prune._validate_pruning_amount_init(amount=0.34)
        prune._validate_pruning_amount_init(amount=1500)
        prune._validate_pruning_amount_init(amount=0)
        prune._validate_pruning_amount_init(amount=0.0)
        prune._validate_pruning_amount_init(amount=1)
        prune._validate_pruning_amount_init(amount=1.0)
        self.assertTrue(True)

    def test_validate_pruning_amount(self):
        r"""Tests the second util function that validates the pruning
        amount requested by the user, this time with respect to the size
        of the tensor to prune. The rationale is that if the pruning amount,
        converted to absolute value of units to prune, is larger than
        the number of units in the tensor, then we expect the util function
        to raise a value error.
        """
        # if amount is int and amount > tensor_size, raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount(amount=20, tensor_size=19)

        # amount is a float so this should not raise an error
        prune._validate_pruning_amount(amount=0.3, tensor_size=0)

        # this is okay
        prune._validate_pruning_amount(amount=19, tensor_size=20)
        prune._validate_pruning_amount(amount=0, tensor_size=0)
        prune._validate_pruning_amount(amount=1, tensor_size=1)
        self.assertTrue(True)

    def test_compute_nparams_to_prune(self):
        r"""Test that requested pruning `amount` gets translated into the
        correct absolute number of units to prune.
        """
        self.assertEqual(prune._compute_nparams_toprune(amount=0, tensor_size=15), 0)
        self.assertEqual(prune._compute_nparams_toprune(amount=10, tensor_size=15), 10)
        # if 1 is int, means 1 unit
        self.assertEqual(prune._compute_nparams_toprune(amount=1, tensor_size=15), 1)
        # if 1. is float, means 100% of units
        self.assertEqual(prune._compute_nparams_toprune(amount=1.0, tensor_size=15), 15)
        self.assertEqual(prune._compute_nparams_toprune(amount=0.4, tensor_size=17), 7)

    def test_random_pruning_sizes(self):
        r"""Test that the new parameters and buffers created by the pruning
        method have the same size as the input tensor to prune. These, in
        fact, correspond to the pruned version of the tensor itself, its
        mask, and its original copy, so the size must match.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    original_tensor = getattr(m, name)

                    prune.random_unstructured(m, name=name, amount=0.1)
                    # mask has the same size as tensor being pruned
                    self.assertEqual(
                        original_tensor.size(), getattr(m, name + "_mask").size()
                    )
                    # 'orig' tensor has the same size as the original tensor
                    self.assertEqual(
                        original_tensor.size(), getattr(m, name + "_orig").size()
                    )
                    # new tensor has the same size as the original tensor
                    self.assertEqual(original_tensor.size(), getattr(m, name).size())

    def test_random_pruning_orig(self):
        r"""Test that original tensor is correctly stored in 'orig'
        after pruning is applied. Important to make sure we don't
        lose info about the original unpruned parameter.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):

                    # tensor prior to pruning
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    result = flow.sum(
                        original_tensor - getattr(m, name + "_orig")
                    ).item()
                    self.assertEqual(result, 0)

    def test_random_pruning_new_weight(self):
        r"""Test that module.name now contains a pruned version of
        the original tensor obtained from multiplying it by the mask.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # tensor prior to pruning
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    # weight = weight_orig * weight_mask
                    weight = getattr(m, name)
                    weight_orig_mask = getattr(m, name + "_orig") * getattr(
                        m, name + "_mask"
                    ).to(dtype=original_tensor.dtype)
                    result = flow.sum(weight - weight_orig_mask).item()

                    self.assertEqual(result, 0)

    def test_identity_pruning(self):
        r"""Test that a mask of 1s does not change forward or backward.
        """
        input_ = flow.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(flow.sum(old_grad_weight - flow.ones_like(m.weight)).item(), 0)
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(flow.sum(old_grad_bias - flow.ones_like(m.bias)).item(), 0)

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        prune.identity(m, name="weight")

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        self.assertEqual(flow.sum(y_prepruning - y_postpruning).item(), 0)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        self.assertEqual(flow.sum(old_grad_weight - m.weight_orig.grad).item(), 0)
        self.assertEqual(flow.sum(old_grad_bias - m.bias.grad).item(), 0)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(flow.sum(y1 - y2).item(), 0)

    def test_random_pruning_0perc(self):
        r"""Test that a mask of 1s does not change forward or backward.
        """
        input_ = flow.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(flow.sum(old_grad_weight - flow.ones_like(m.weight)).item(), 0)
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(flow.sum(old_grad_bias - flow.ones_like(m.bias)).item(), 0)

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        with mock.patch(
            "oneflow.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = flow.ones_like(m.weight)
            prune.random_unstructured(
                m, name="weight", amount=0.9
            )  # amount won't count

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        self.assertEqual(flow.sum(y_prepruning - y_postpruning).item(), 0)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        self.assertEqual(flow.sum(old_grad_weight - m.weight_orig.grad).item(), 0)
        self.assertEqual(flow.sum(old_grad_bias - m.bias.grad).item(), 0)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(flow.sum(y1 - y2).item(), 0)

    def test_random_pruning(self):
        input_ = flow.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = flow.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "oneflow.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name="weight", amount=0.9)

        y_postpruning = m(input_)
        y_postpruning.sum().backward()
        # weight_orig is the parameter, so it's the tensor that will accumulate the grad
        self.assertEqual(
            flow.sum(m.weight_orig.grad - mask).item(), 0
        )  # all 1s, except for masked units
        self.assertEqual(flow.sum(m.bias.grad - flow.ones_like(m.bias)).item(), 0)

        # make sure that weight_orig update doesn't modify [1, 0] and [0, 3]
        old_weight_orig = m.weight_orig.clone()
        # update weights
        learning_rate = 1.0
        for p in m.parameters():
            p.data.sub_(p.grad.data * learning_rate)
        # since these are pruned, they should not be updated
        self.assertEqual(
            flow.sum(old_weight_orig[1, 0] - m.weight_orig[1, 0]).item(), 0
        )
        self.assertEqual(
            flow.sum(old_weight_orig[0, 3] - m.weight_orig[0, 3]).item(), 0
        )

    def test_random_pruning_forward(self):
        r"""check forward with mask (by hand).
        """
        input_ = flow.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = flow.zeros_like(m.weight)
        mask[1, 0] = 1
        mask[0, 3] = 1

        with mock.patch(
            "oneflow.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name="weight", amount=0.9)

        yhat = m(input_)
        self.assertTrue(
            flow.sum(yhat[0, 0] - m.weight_orig[0, 3] - m.bias[0]).item() - 0 < 1e-5
        )
        self.assertTrue(
            flow.sum(yhat[0, 1] - m.weight_orig[1, 0] - m.bias[1]).item() - 0 < 1e-5
        )

    def test_remove_pruning_forward(self):
        r"""Remove pruning and check forward is unchanged from previous
        pruned state.
        """
        input_ = flow.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = flow.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "oneflow.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name="weight", amount=0.9)

        y_postpruning = m(input_)

        prune.remove(m, "weight")

        y_postremoval = m(input_)
        self.assertEqual(flow.sum(y_postpruning - y_postremoval).item(), 0)

    def test_pruning_id_consistency(self):
        r"""Test that pruning doesn't change the id of the parameters, which
        would otherwise introduce issues with pre-existing optimizers that
        point to old parameters.
        """
        m = nn.Linear(5, 2, bias=False)

        tensor_id = id(list(m.parameters())[0])

        prune.random_unstructured(m, name="weight", amount=0.9)
        self.assertEqual(tensor_id, id(list(m.parameters())[0]))

        prune.remove(m, "weight")
        self.assertEqual(tensor_id, id(list(m.parameters())[0]))

    def test_random_pruning_pickle(self):
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    prune.random_unstructured(m, name=name, amount=0.1)
                    m_new = pickle.loads(pickle.dumps(m))
                    self.assertIsInstance(m_new, type(m))

    def test_multiple_pruning_calls(self):
        # if you call pruning twice, the hook becomes a PruningContainer
        m = nn.Conv3d(2, 2, 2)
        prune.l1_unstructured(m, name="weight", amount=0.1)
        weight_mask0 = m.weight_mask  # save it for later sanity check

        # prune again
        prune.ln_structured(m, name="weight", amount=0.3, n=2, dim=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertIsInstance(hook, oneflow.nn.utils.prune.PruningContainer)
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        self.assertEqual(hook._tensor_name, "weight")

        # check that the pruning container has the right length
        # equal to the number of pruning iters
        self.assertEqual(len(hook), 2)  # m.weight has been pruned twice

        # check that the entries of the pruning container are of the expected
        # type and in the expected order
        self.assertIsInstance(hook[0], oneflow.nn.utils.prune.L1Unstructured)
        self.assertIsInstance(hook[1], oneflow.nn.utils.prune.LnStructured)

        # check that all entries that are 0 in the 1st mask are 0 in the
        # 2nd mask too
        self.assertTrue(flow.all(m.weight_mask[weight_mask0 == 0] == 0))

        # prune again
        prune.ln_structured(m, name="weight", amount=0.1, n=float("inf"), dim=1)
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertEqual(hook._tensor_name, "weight")

    def test_pruning_container(self):
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = "test"
        self.assertEqual(len(container), 0)

        p = prune.L1Unstructured(amount=2)
        p._tensor_name = "test"

        # test adding a pruning method to a container
        container.add_pruning_method(p)

        # test error raised if tensor name is different
        q = prune.L1Unstructured(amount=2)
        q._tensor_name = "another_test"
        with self.assertRaises(ValueError):
            container.add_pruning_method(q)

        # test that adding a non-pruning method object to a pruning container
        # raises a TypeError
        with self.assertRaises(TypeError):
            container.add_pruning_method(10)
        with self.assertRaises(TypeError):
            container.add_pruning_method("ugh")

    def test_pruning_container_compute_mask(self):
        r"""Test `compute_mask` of pruning container with a known `t` and
        `default_mask`. Indirectly checks that Ln structured pruning is
        acting on the right axis.
        """
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = "test"

        # 1) test unstructured pruning
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = "test"
        # add the pruning method to the container
        container.add_pruning_method(p)

        # create tensor to be pruned
        t = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=flow.float32)
        # create prior mask by hand
        default_mask = flow.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = flow.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=flow.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(flow.sum(expected_mask - computed_mask).item(), 0)

        # 2) test structured pruning
        q = prune.LnStructured(amount=1, n=2, dim=0)
        q._tensor_name = "test"
        container.add_pruning_method(q)
        # since we are pruning the lowest magnitude one of the two rows, the
        # outcome of the calculation should be this:
        expected_mask = flow.tensor([[0, 0, 0, 0], [1, 1, 0, 1]], dtype=flow.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(flow.sum(expected_mask - computed_mask).item(), 0)

        # 2) test structured pruning, along another axis
        r = prune.LnStructured(amount=1, n=2, dim=1)
        r._tensor_name = "test"
        container.add_pruning_method(r)
        # since we are pruning the lowest magnitude of the four columns, the
        # outcome of the calculation should be this:
        expected_mask = flow.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], dtype=flow.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(flow.sum(expected_mask - computed_mask).item(), 0)

    def test_l1_unstructured_pruning(self):
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries by l1 norm (by hand). It also checks that applying l1
        unstructured pruning more than once respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = flow.nn.Parameter(
            flow.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=flow.float32)
        )

        prune.l1_unstructured(m, "weight", amount=2)
        expected_weight = flow.tensor(
            [[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(flow.sum(expected_weight - m.weight).item(), 0)

        # check that pruning again removes the next two smallest entries
        prune.l1_unstructured(m, "weight", amount=2)
        expected_weight = flow.tensor(
            [[0, 0, 3, 4], [-4, -3, 0, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(flow.sum(expected_weight - m.weight).item(), 0)

    def test_l1_unstructured_pruning_with_importance_scores(self):
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries of importance scores and not the parameter by l1 norm (by hand).
        It also checks that applying l1 unstructured pruning more than once
        respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = flow.nn.Parameter(
            flow.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=flow.float32)
        )
        importance_scores = flow.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=flow.float32
        )

        prune.l1_unstructured(
            m, "weight", amount=2, importance_scores=importance_scores
        )
        expected_weight = flow.tensor(
            [[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(flow.sum(expected_weight - m.weight).item(), 0)

        # check that pruning again removes two entries of m.weight that are colocated with
        # the next two smallest absolute values of importance scores.
        prune.l1_unstructured(
            m, "weight", amount=2, importance_scores=importance_scores
        )
        expected_weight = flow.tensor(
            [[1, 0, 0, 4], [-4, 0, 0, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(flow.sum(expected_weight - m.weight).item(), 0)

    def test_unstructured_pruning_same_magnitude(self):
        r"""Since it may happen that the tensor to prune has entries with the
        same exact magnitude, it is important to check that pruning happens
        consistenly based on the bottom % of weights, and not by threshold,
        which would instead kill off *all* units with magnitude = threshold.
        """
        AMOUNT = 0.2
        p = prune.L1Unstructured(amount=AMOUNT)
        # create a random tensors with entries in {-2, 0, 2}
        t = 2 * flow.randint(low=-1, high=2, size=(10, 7))
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.nelement())

        computed_mask = p.compute_mask(t, default_mask=flow.ones_like(t))
        nparams_pruned = flow.sum(computed_mask == 0)
        self.assertEqual(nparams_toprune, nparams_pruned)

    def test_random_structured_pruning_amount(self):
        AMOUNT = 0.6
        AXIS = 2
        p = prune.RandomStructured(amount=AMOUNT, dim=AXIS)
        t = 2 * flow.randint(low=-1, high=2, size=(5, 4, 2)).to(dtype=flow.float32)
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.shape[AXIS])

        computed_mask = p.compute_mask(t, default_mask=flow.ones_like(t))
        # check that 1 column is fully prune, the others are left untouched
        remaining_axes = [_ for _ in range(len(t.shape)) if _ != AXIS]
        per_column_sums = sorted(flow.sum(computed_mask == 0, dim=remaining_axes))
        assert per_column_sums == [0, 20]

    def test_ln_structured_pruning(self):
        r"""Check Ln structured pruning by hand.
        """
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = flow.tensor(
            [
                [
                    [[1.0, 2.0], [1.0, 2.5]],
                    [[0.5, 1.0], [0.1, 0.1]],
                    [[-3.0, -5.0], [0.1, -1.0]],
                ]
            ]
        )
        # expected effect of pruning 1 of the 3 channels by L2-norm
        expected_mask_axis1 = flow.ones_like(m.weight)
        expected_mask_axis1[:, 1] = 0.0

        prune.ln_structured(m, "weight", amount=1, n=2, dim=1)
        self.assertEqual(flow.sum(expected_mask_axis1 - m.weight_mask).item(), 0)

        # expected effect of pruning 1 of the 2 columns along axis -1 by L1-norm
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 0] = 0.0

        prune.ln_structured(m, "weight", amount=1, n=1, dim=-1)
        self.assertEqual(flow.sum(expected_mask_axis3 - m.weight_mask).item(), 0)

    def test_ln_structured_pruning_importance_scores(self):
        r"""Check Ln structured pruning by hand.
        """
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = flow.tensor(
            [
                [
                    [[1.0, 2.0], [1.0, 2.5]],
                    [[0.5, 1.0], [0.1, 0.1]],
                    [[-3.0, -5.0], [0.1, -1.0]],
                ]
            ]
        )
        importance_scores = flow.tensor(
            [
                [
                    [[10.0, 1.0], [10.0, 1.0]],
                    [[30.0, 3.0], [30.0, 3.0]],
                    [[-20.0, -2.0], [-20.0, -2.0]],
                ]
            ]
        )
        # expected effect of pruning 1 of the 3 channels by L2-norm
        expected_mask_axis1 = flow.ones_like(m.weight)
        expected_mask_axis1[:, 0] = 0.0

        prune.ln_structured(
            m, "weight", amount=1, n=2, dim=1, importance_scores=importance_scores
        )
        self.assertEqual(flow.sum(expected_mask_axis1 - m.weight_mask).item(), 0)

        # expected effect of pruning 1 of the 2 columns along axis -1 by L1-norm
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 1] = 0.0

        prune.ln_structured(
            m, "weight", amount=1, n=1, dim=-1, importance_scores=importance_scores
        )
        self.assertEqual(flow.sum(expected_mask_axis3 - m.weight_mask).item(), 0)

    def test_remove_pruning(self):
        r"""`prune.remove` removes the hook and the reparametrization
        and makes the pruning final in the original parameter.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # first prune
                    prune.random_unstructured(m, name, amount=0.5)
                    self.assertIn(name + "_orig", dict(m.named_parameters()))
                    self.assertIn(name + "_mask", dict(m.named_buffers()))
                    self.assertNotIn(name, dict(m.named_parameters()))
                    self.assertTrue(hasattr(m, name))
                    pruned_t = getattr(m, name)

                    # then remove pruning
                    prune.remove(m, name)
                    self.assertIn(name, dict(m.named_parameters()))
                    self.assertNotIn(name + "_orig", dict(m.named_parameters()))
                    self.assertNotIn(name + "_mask", dict(m.named_buffers()))
                    final_t = getattr(m, name)

                    self.assertEqual(flow.sum(pruned_t - final_t).item(), 0)

    def test_remove_pruning_exception(self):
        r"""Removing from an unpruned tensor throws an assertion error
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # check that the module isn't pruned
                    self.assertFalse(prune.is_pruned(m))
                    # since it isn't pruned, pruning can't be removed from it
                    with self.assertRaises(ValueError):
                        prune.remove(m, name)

    def test_global_pruning(self):
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # modify the weight matrices by hand
        m.weight = flow.nn.Parameter(
            flow.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=flow.float32)
        )
        n.weight = flow.nn.Parameter(flow.tensor([[0, 0.1, -2]]).to(dtype=flow.float32))

        params_to_prune = (
            (m, "weight"),
            (n, "weight"),
        )

        # prune the 4 smallest weights globally by L1 magnitude
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=4
        )

        expected_mweight = flow.tensor(
            [[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(flow.sum(expected_mweight - m.weight).item(), 0)

        expected_nweight = flow.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        self.assertEqual(flow.sum(expected_nweight - n.weight).item(), 0)

    def test_global_pruning_importance_scores(self):
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # modify the weight matrices by hand
        m.weight = flow.nn.Parameter(
            flow.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=flow.float32)
        )
        m_importance_scores = flow.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=flow.float32
        )
        n.weight = flow.nn.Parameter(flow.tensor([[0, 0.1, -2]]).to(dtype=flow.float32))
        n_importance_scores = flow.tensor([[0, 10.0, -0.2]]).to(dtype=flow.float32)

        params_to_prune = (
            (m, "weight"),
            (n, "weight"),
        )
        importance_scores = {
            (m, "weight"): m_importance_scores,
            (n, "weight"): n_importance_scores,
        }

        # prune the 4 smallest weights globally by L1 magnitude
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=4,
            importance_scores=importance_scores,
        )

        expected_m_weight = flow.tensor(
            [[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(flow.sum(expected_m_weight - m.weight).item(), 0)

        expected_n_weight = flow.tensor([[0, 0.1, 0]]).to(dtype=n.weight.dtype)
        self.assertEqual(flow.sum(expected_n_weight - n.weight).item(), 0)

    def test_custom_from_mask_pruning(self):
        r"""Test that the CustomFromMask is capable of receiving
        as input at instantiation time a custom mask, and combining it with
        the previous default mask to generate the correct final mask.
        """
        # new mask
        mask = flow.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
        # old mask
        default_mask = flow.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

        # some tensor (not actually used)
        t = flow.rand(mask.shape, dtype=flow.float32, device=mask.device)
        # t = flow.rand_like(mask.to(dtype=flow.float32))

        p = prune.CustomFromMask(mask=mask)

        computed_mask = p.compute_mask(t, default_mask)
        expected_mask = flow.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 1]], dtype=computed_mask.dtype
        )

        self.assertEqual(flow.sum(computed_mask - expected_mask).item(), 0)

    def test_pruning_rollback(self):
        r"""Test that if something fails when the we try to compute the mask,
        then the model isn't left in some intermediate half-pruned state.
        The try/except statement in `apply` should handle rolling back
        to the previous state before pruning began.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):

                    with mock.patch(
                        "oneflow.nn.utils.prune.L1Unstructured.compute_mask"
                    ) as compute_mask:
                        compute_mask.side_effect = Exception("HA!")
                        with self.assertRaises(Exception):
                            prune.l1_unstructured(m, name=name, amount=0.9)

                        self.assertTrue(name in dict(m.named_parameters()))
                        self.assertFalse(name + "_mask" in dict(m.named_buffers()))
                        self.assertFalse(name + "_orig" in dict(m.named_parameters()))

    def test_pruning_serialization_model(self):
        # create a model
        model = flow.nn.Sequential(
            flow.nn.Linear(10, 10), flow.nn.ReLU(), flow.nn.Linear(10, 1),
        )
        # check that everything looks normal before pruning
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # prune one of its parameters
        prune.l1_unstructured(module=model[0], name="weight", amount=0.9)

        # check that the original weight and the new mask are present
        self.assertIn("0.weight_orig", model.state_dict())
        self.assertIn("0.weight_mask", model.state_dict())
        self.assertNotIn("0.weight", model.state_dict())
        self.assertTrue(hasattr(model[0], "weight"))

        pruned_weight = model[0].weight

        with tempfile.NamedTemporaryFile() as f:
            flow.save(model, f.name)
            new_model = flow.load(f.name)

        # check that the original weight and the new mask are present
        self.assertIn("0.weight_orig", new_model.state_dict())
        self.assertIn("0.weight_mask", new_model.state_dict())
        self.assertNotIn("0.weight", new_model.state_dict())
        self.assertTrue(hasattr(new_model[0], "weight"))

        self.assertEqual(flow.sum(pruned_weight - new_model[0].weight).item(), 0)

    def test_prune(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=flow.float32)
        # create prior mask by hand
        default_mask = flow.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = flow.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask)
        self.assertEqual(flow.sum(t * expected_mask - pruned_tensor).item(), 0)

    def test_prune_importance_scores(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=flow.float32)
        importance_scores = flow.tensor([[1, 2, 3, 4], [1.5, 1.6, 1.7, 1.8]]).to(
            dtype=flow.float32
        )
        # create prior mask by hand
        default_mask = flow.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = flow.tensor([[0, 1, 1, 0], [0, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask, importance_scores=importance_scores)
        self.assertEqual(flow.sum(t * expected_mask - pruned_tensor).item(), 0)

    def test_prune_importance_scores_mimic_default(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=flow.float32)
        # create prior mask by hand
        default_mask = flow.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = flow.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor_without_importance_scores = p.prune(t, default_mask)
        pruned_tensor_with_importance_scores = p.prune(
            t, default_mask, importance_scores=t
        )
        self.assertEqual(
            flow.sum(
                pruned_tensor_without_importance_scores
                - pruned_tensor_with_importance_scores
            ).item(),
            0,
        )
        self.assertEqual(
            flow.sum(
                t * expected_mask - pruned_tensor_without_importance_scores
            ).item(),
            0,
        )

    def test_rnn_pruning(self):
        l = flow.nn.LSTM(32, 32)
        # This Module has 4 parameters called:
        # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'

        # Pruning one of them causes one of the weights to become a tensor
        prune.l1_unstructured(l, "weight_ih_l0", 0.5)
        assert sum([isinstance(p, flow.nn.Parameter) for p in l._flat_weights]) == 3

        # Removing the pruning reparametrization restores the Parameter
        prune.remove(l, "weight_ih_l0")
        assert sum([isinstance(p, flow.nn.Parameter) for p in l._flat_weights]) == 4

        # Make sure that, upon removal of the reparametrization, the
        # `._parameters` and `.named_parameters` contain the right params.
        # Specifically, the original weight ('weight_ih_l0') should be placed
        # back in the parameters, while the reparametrization component
        # ('weight_ih_l0_orig') should be removed.
        assert "weight_ih_l0" in l._parameters
        assert l._parameters["weight_ih_l0"] is not None
        assert "weight_ih_l0_orig" not in l._parameters
        assert "weight_ih_l0" in dict(l.named_parameters())
        assert dict(l.named_parameters())["weight_ih_l0"] is not None
        assert "weight_ih_l0_orig" not in dict(l.named_parameters())


if __name__ == "__main__":
    unittest.main()
