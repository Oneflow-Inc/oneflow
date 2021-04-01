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

import numpy as np

import oneflow as flow
import oneflow.typing as tp


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_logsoftmax(test_case):
        m1 = flow.nn.LogSoftmax(dim=1)
        x1 = flow.Tensor(
            np.array(
                [[ 0.4296, -1.1957,  2.5463],
                [ 1.2552, -1.5747,  0.6923]]
            )
        )
        y1 = m1(x1)
        torch_out1 = np.array(
            [[-2.25134873, -3.87664890, -0.13464880],
            [-0.48770463, -3.31760454, -1.05060458]]
        )
        print(np.allclose(y1.numpy(), torch_out1, rtol=1e-05))



        m2 = flow.nn.LogSoftmax(dim=2)
        x2 = flow.Tensor(
            np.array(
                [[[[ 2.55630851,  1.57471120,  0.25240266, -0.57634264],
                [ 0.72222596,  0.35014620,  0.43715513,  1.41162395],
                [ 0.12103304, -1.15901530,  1.08098269,  0.04042318]],

                [[-0.43854573,  1.07273626,  1.80571628, -1.72897887],
                [ 1.33143651, -1.53906214, -1.45766914, -1.10325944],
                [-0.14271064,  0.38371494,  0.45469931, -0.32640621]]],


                [[[ 1.75221181, -1.04841065, -0.41832924, -0.56028533],
                [ 0.89104170, -0.10217502, -0.21890916, -0.24784143],
                [-0.35868922, -1.64597833, -0.19254614,  0.24511036]],

                [[ 0.08831860,  2.37159801,  0.79408669, -0.39868262],
                [-0.69181246, -1.11924624, -0.47067565,  0.14795294],
                [-0.23413812, -0.71034479,  0.44405901, -0.90119874]]]]
            )
        )

        y2 = m2(x2)
        torch_out2 = np.array(
            [[[[-0.22100820, -0.30664772, -1.50251734, -2.31782818],
            [-2.05509090, -1.53121281, -1.31776488, -0.32986164],
            [-2.65628386, -3.04037428, -0.67393732, -1.70106244]],

            [[-2.10596132, -0.45455331, -0.26023546, -1.93661511],
            [-0.33597916, -3.06635165, -3.52362084, -1.31089568],
            [-1.81012630, -1.14357471, -1.61125243, -0.53404248]]],


            [[[-0.43424436, -1.41734290, -1.24530625, -1.52699995],
            [-1.29541445, -0.47110727, -1.04588616, -1.21455598],
            [-2.54514551, -2.01491070, -1.01952314, -0.72160423]],

            [[-0.78056860, -0.07357123, -0.68661338, -1.20370412],
            [-1.56069970, -3.56441545, -1.95137572, -0.65706855],
            [-1.10302532, -3.15551400, -1.03664112, -1.70622015]]]]
        )

        print(np.allclose(y2.numpy(), torch_out2, rtol=1e-05))



if __name__ == "__main__":
    unittest.main()
