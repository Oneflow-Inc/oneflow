from oneflow.compatible import single_client as flow
import env_1node
import os
from absl import app
from absl.testing import absltest
from test_1node_mixin import Test1NodeMixin
from cnns_tests import (
    TestAlexNetMixin,
    TestResNet50Mixin,
    TestVgg16Mixin,
    TestInceptionV3Mixin,
)


class TestAlexNet(Test1NodeMixin, TestAlexNetMixin, absltest.TestCase):
    def setUp(self):
        super().setUp()
        flow.enable_eager_execution(True)


class TestResNet50(Test1NodeMixin, TestResNet50Mixin, absltest.TestCase):
    def setUp(self):
        super().setUp()
        flow.enable_eager_execution(True)


class TestVgg16(Test1NodeMixin, TestVgg16Mixin, absltest.TestCase):
    def setUp(self):
        super().setUp()
        flow.enable_eager_execution(True)


class TestInceptionV3(Test1NodeMixin, TestInceptionV3Mixin, absltest.TestCase):
    def setUp(self):
        super().setUp()
        flow.enable_eager_execution(True)


class TestEagerMixin(object):
    def setUp(self):
        flow.clear_default_session()
        flow.enable_eager_execution(True)


flow.unittest.register_test_cases(
    scope=globals(),
    directory=os.path.dirname(os.path.realpath(__file__)),
    filter_by_num_nodes=lambda x: x == 1,
    base_class=absltest.TestCase,
    test_case_mixin=TestEagerMixin,
)


def main(argv):
    env_1node.Init()
    absltest.main()


if __name__ == "__main__":
    app.run(main)
