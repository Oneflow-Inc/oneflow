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
from collections import OrderedDict

import numpy as np
import oneflow as flow
from oneflow.test_utils.test_util import GenArgList
from sklearn.metrics import roc_auc_score


def _test_roc_auc_score(test_case, label_dtype, pred_dtype):
    inputs = [
        {"label": [0, 0, 1, 1], "pred": [0.1, 0.4, 0.35, 0.8], "score": 0.75},
        {"label": [0, 1, 0, 1], "pred": [0.5, 0.5, 0.5, 0.5], "score": 0.5},
    ]
    for data in inputs:
        label = flow.tensor(data["label"], dtype=label_dtype)
        pred = flow.tensor(data["pred"], dtype=pred_dtype)
        of_score = flow.roc_auc_score(label, pred)
        test_case.assertTrue(np.allclose(of_score.numpy()[0], data["score"]))


def _compare_roc_auc_score(test_case, label_dtype, pred_dtype):
    n_examples = 16384
    label = np.random.randint(0, 2, n_examples)
    pred = np.random.random(n_examples)
    score = roc_auc_score(label, pred)

    label = flow.tensor(label, dtype=label_dtype)
    pred = flow.tensor(pred, dtype=pred_dtype)
    of_score = flow.roc_auc_score(label, pred)

    test_case.assertTrue(np.allclose(of_score.numpy()[0], score))


@flow.unittest.skip_unless_1n1d()
class TestNMS(flow.unittest.TestCase):
    def test_roc_auc_score(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_roc_auc_score, _compare_roc_auc_score]
        arg_dict["label_dtype"] = [
            flow.double,
            flow.int32,
            flow.float,
            flow.int64,
            flow.int8,
            flow.uint8,
        ]
        arg_dict["pred_dtype"] = [flow.float]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
