from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('nodes_list', '192.168.1.15,192.168.1.14', 'nodes list seperated by comma')

class Test2NodeMixin:
  def test_2n8c(self):
    self.run_net(4, 2, FLAGS.nodes_list)
    self.print_and_check_result('2n8c')
