import os
import numpy
from absl import app
from absl.testing import absltest
import env_1node
import cnns_tests
from test_1node_mixin import Test1NodeMixin
import oneflow as flow

class TestAlexNet(Test1NodeMixin, cnns_tests.TestAlexNetMixin, absltest.TestCase):
  pass

class TestResNet50(Test1NodeMixin, cnns_tests.TestResNet50Mixin, absltest.TestCase):
  pass

class TestVgg16(Test1NodeMixin, cnns_tests.TestVgg16Mixin, absltest.TestCase):
  pass

class TestInceptionV3(Test1NodeMixin, cnns_tests.TestInceptionV3Mixin, absltest.TestCase):
  pass

flow.unittest.register_test_cases(
    scope=globals(),
    directory=os.path.dirname(os.path.realpath(__file__)),
    filter_by_num_nodes=lambda x: x == 1,
    base_class=absltest.TestCase)

def main(argv):
  env_1node.Init()
  absltest.main()

if __name__ == '__main__': app.run(main)
