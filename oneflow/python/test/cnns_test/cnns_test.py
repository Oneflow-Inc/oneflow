import unittest

class TestNet(unittest.TestCase):
  """
    Base Tester
  """
  def build_net(self):
    """
    build a test network
    @return
      DLNet, the built network
    """
    pass

  def set_up_param(self):
    """
    set up params for test network
    @return
      None
    """
    pass

  def test_1n1c(self):
    pass

  def test_1n4c(self):
    pass

  def test_2n8c(self):
    pass

  def test_report(self):
    """
    ======================================================================
    result report
    ======================================================================
    of-1n1c -----------------------------------------------------     pass
    of-1n4c ----------------------------------------------------- not pass
    of-2n4c -----------------------------------------------------     pass

    ======================================================================
    xx net loss report
    ======================================================================
    iter     tf          of-1n1c      of-1n4c       of-2n4c
    0        6.932688    6.932688     6.932688      6.932688
    1        6.924820    ...          ...           ...
    2        6.917069
    3        6.909393
    4        6.901904
    5        6.894367
    6        6.886764
    7        6.879305
    8        6.872003
    9        6.864939

    """

    pass


class TestAlexNet(TestNet):
  """
    AlexNet Tester
  """
  def buildNet(self):
    pass

  def set_up_param(self):
    pass


class TestVgg16Net(TestNet):
  """
    AlexNet Tester
  """
  def buildNet(self):
    pass

  def set_up_param(self):
    pass



if __name__ == '__main__':
  unittest.main()
