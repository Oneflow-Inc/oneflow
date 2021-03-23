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
import numpy
from absl import app
from absl.testing import absltest
import oneflow as flow

flow.unittest.register_test_cases(
    scope=globals(),
    directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "nodes/pytorch"
    ),
    filter_by_num_nodes=lambda x: True,
    base_class=absltest.TestCase,
)


def main(argv):
    flow.env.init()
    absltest.main()


if __name__ == "__main__":
    app.run(main)
