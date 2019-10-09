import os
import unittest
import numpy

NODE_LIST = "192.168.1.12,192.168.1.14"

class TestNet(unittest.TestCase):
  """
    Base Tester
  """
  def setUp(self):
    self.net = ''
    self.tf_loss_dir = ''
    self.of_loss_dir = ''
    self.num_iter = 10
    self.set_params()

  def set_params(self):
    pass

  def run_net(self, num_gpu_per_node, num_node = 1, node_list = ""):
    if num_node > 1:
      os.system("python {}.py -g {} -m -n {}".format(self.net, num_gpu_per_node, node_list))
    else:
      os.system("python {}.py -g {}".format(self.net, num_gpu_per_node))

  def load_tf_loss(self):
    tf_loss = numpy.load(os.path.join(self.tf_loss_dir, '1n1c.npy'))
    return tf_loss[0:self.num_iter]

  def load_of_loss(self, type):
    of_loss = numpy.load(os.path.join(self.of_loss_dir, type + '.npy'))
    return of_loss[0:self.num_iter]

  def run_and_compare(self, num_gpu_per_node, num_node = 1, node_list = ""):
    self.run_net(num_gpu_per_node, num_node, node_list)
    tf_loss = self.load_tf_loss()
    of_loss = self.load_of_loss('{}n{}c'.format(num_node, num_gpu_per_node * num_node))
    self.assertTrue(numpy.allclose(tf_loss, of_loss, atol=1e-5), "Compare not pass!")

  def print_report(self):
    loss_dict = {}
    loss_dict['tf'] = self.load_tf_loss()
    loss_dict['of_1n1c'] = self.load_of_loss('1n1c')
    loss_dict['of_1n4c'] = self.load_of_loss('1n4c')
    loss_dict['of_2n8c'] = self.load_of_loss('2n8c')

    print("==".ljust(64, '='))
    print(" ".ljust(2, ' ') + self.net + " loss report")
    print("==".ljust(64, '='))
    fmt_str = "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}"
    print(fmt_str.format("iter", "tf", "of_1n1c", "of_1n4c", "of_2n8c"))
    for i in range(self.num_iter):
      fmt_str = "{:>6}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}"
      print(fmt_str.format(i,loss_dict['tf'][i], loss_dict['of_1n1c'][i],
                           loss_dict['of_1n4c'][i], loss_dict['of_2n8c'][i]))


class TestAlexNet(TestNet):
  """
    AlexNet Tester
  """
  def set_params(self):
    self.net = 'alexnet'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

  def test_1n1c(self):
    self.run_and_compare(1)

  def test_1n4c(self):
    self.run_and_compare(4)

  def test_2n8c(self):
    self.run_and_compare(4, 2, NODE_LIST)

  def test_report(self):
    self.print_report()


class TestResNet50(TestNet):
  """
    AlexNet Tester
  """
  def set_params(self):
    self.net = 'resnet50'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

  def test_1n1c(self):
    self.run_and_compare(1)

  def test_1n4c(self):
    self.run_and_compare(4)

  def test_2n8c(self):
    self.run_and_compare(4, 2, NODE_LIST)

  def test_report(self):
    self.print_report()


class TestVgg16(TestNet):
  """
    Vgg16 Tester
  """
  def set_params(self):
    self.net = 'vgg16'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

  def test_1n1c(self):
    self.run_and_compare(1)

  def test_1n4c(self):
    self.run_and_compare(4)

  def test_2n8c(self):
    self.run_and_compare(4, 2, NODE_LIST)

  def test_report(self):
    self.print_report()


class TestInceptionV3(TestNet):
  """
    InceptionV3 Tester
  """
  def set_params(self):
    self.net = 'inceptionv3'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

  def test_1n1c(self):
    self.run_and_compare(1)

  def test_1n4c(self):
    self.run_and_compare(4)

  def test_2n8c(self):
    self.run_and_compare(4, 2, NODE_LIST)

  def test_report(self):
    self.print_report()

if __name__ == '__main__':
  unittest.main()
