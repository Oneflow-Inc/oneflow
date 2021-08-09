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

import env_1node
from absl import app
from absl.testing import absltest
from cnns_tests import (
    TestAlexNetMixin,
    TestInceptionV3Mixin,
    TestResNet50Mixin,
    TestVgg16Mixin,
)
from test_1node_mixin import Test1NodeMixin

from oneflow.compatible import single_client as flow


class TestAlexNet(Test1NodeMixin, TestAlexNetMixin, absltest.TestCase):
    pass


class TestResNet50(Test1NodeMixin, TestResNet50Mixin, absltest.TestCase):
    pass


class TestVgg16(Test1NodeMixin, TestVgg16Mixin, absltest.TestCase):
    pass


class TestInceptionV3(Test1NodeMixin, TestInceptionV3Mixin, absltest.TestCase):
    pass


flow.unittest.register_test_cases(
    scope=globals(),
    directory=os.path.dirname(os.path.realpath(__file__)),
    filter_by_num_nodes=lambda x: x == 1,
    base_class=absltest.TestCase,
)


def main(argv):
    env_1node.Init()
    absltest.main()


if __name__ == "__main__":
    app.run(main)
