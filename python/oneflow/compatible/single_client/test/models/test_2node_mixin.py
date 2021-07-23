from absl import flags

FLAGS = flags.FLAGS


class Test2NodeMixin:
    def test_2n8c(self):
        self.run_net(4, 2, FLAGS.nodes_list)
        self.print_and_check_result("2n8c")
