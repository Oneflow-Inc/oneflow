import os
import numpy
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('python_bin', 'python3', 'python binary program name or filepath.')

class TestNetMixin:
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
      os.system("{} {}.py -g {} -m -n {}".format(FLAGS.python_bin, self.net, num_gpu_per_node, node_list))
    else:
      os.system("{} {}.py -g {}".format(FLAGS.python_bin, self.net, num_gpu_per_node))

  def load_tf_loss(self):
    tf_loss = numpy.load(os.path.join(self.tf_loss_dir, '1n1c.npy'))
    return tf_loss[0:self.num_iter]

  def load_of_loss(self, test_type):
    path = os.path.join(self.of_loss_dir, test_type + '.npy')
    if os.path.exists(path):
      of_loss = numpy.load(path)
    else:
      of_loss = numpy.zeros(self.num_iter)
    return of_loss[0:self.num_iter]

  def run_and_compare(self, num_gpu_per_node, num_node = 1, node_list = ""):
    self.run_net(num_gpu_per_node, num_node, node_list)
    tf_loss = self.load_tf_loss()
    of_loss = self.load_of_loss('{}n{}c'.format(num_node, num_gpu_per_node * num_node))
    self.assertTrue(numpy.allclose(tf_loss, of_loss, atol=1e-5), "Compare not pass!")

  def print_and_check_result(self, result_name):
    loss_dict = {}
    loss_dict['tensorflow'] = self.load_tf_loss()
    loss_dict['oneflow'] = self.load_of_loss(result_name)

    print("==".ljust(64, '='))
    print(" ".ljust(2, ' ') + self.net + " loss report")
    print("==".ljust(64, '='))
    fmt_str = "{:>6}  {:>12}  {:>12}"
    print(fmt_str.format("iter", "tensorflow", "oneflow-" + result_name))
    for i in range(self.num_iter):
      fmt_str = "{:>6}  {:>12.6f}  {:>12.6f}"
      print(fmt_str.format(i,loss_dict['tensorflow'][i], loss_dict['oneflow'][i]))
    self.assertTrue(numpy.allclose(loss_dict['tensorflow'], loss_dict['oneflow']))

class TestAlexNetMixin(TestNetMixin):
  """
    AlexNet Tester
  """
  def set_params(self):
    self.net = 'alexnet'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

class TestResNet50Mixin(TestNetMixin):
  """
    AlexNet Tester
  """
  def set_params(self):
    self.net = 'resnet50'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

class TestVgg16Mixin(TestNetMixin):
  """
    Vgg16 Tester
  """
  def set_params(self):
    self.net = 'vgg16'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)

class TestInceptionV3Mixin(TestNetMixin):
  """
    InceptionV3 Tester
  """
  def set_params(self):
    self.net = 'inceptionv3'
    self.tf_loss_dir = os.path.join("/dataset/PNGS/cnns_model_for_test/tf_loss", self.net)
    self.of_loss_dir = os.path.join("./of_loss", self.net)
