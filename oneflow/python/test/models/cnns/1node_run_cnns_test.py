import os
import numpy
from absl import app
from absl.testing import absltest
import cnns_tests
from test_1node_mixin import Test1NodeMixin

class TestAlexNet(Test1NodeMixin, cnns_tests.TestAlexNetMixin, absltest.TestCase):
  pass

class TestResNet50(Test1NodeMixin, cnns_tests.TestResNet50Mixin, absltest.TestCase):
  pass

class TestVgg16(Test1NodeMixin, cnns_tests.TestVgg16Mixin, absltest.TestCase):
  pass

class TestInceptionV3(Test1NodeMixin, cnns_tests.TestInceptionV3Mixin, absltest.TestCase):
  pass

if __name__ == '__main__':
  app.run(lambda argv: absltest.main())
