import os
from absl import app
from absl.testing import absltest
import oneflow as flow

from absl import flags
import atexit

FLAGS = flags.FLAGS
flags.DEFINE_string('nodes_list', '192.168.1.15,192.168.1.16', 'nodes list seperated by comma')
flags.DEFINE_integer('ctrl_port', '9524', 'control port')

def Init():
  flow.env.machine(FLAGS.nodes_list.split(','))
  flow.env.ctrl_port(FLAGS.ctrl_port)
  flow.deprecated.init_worker(scp_binary=True, use_uuid=True)
  atexit.register(flow.deprecated.delete_worker)

flow.unittest.register_test_cases(
    scope=globals(),
    directory=os.path.dirname(os.path.realpath(__file__)),
    filter_by_num_nodes=lambda x: x == 2,
    base_class=absltest.TestCase)

def main(argv):
  Init()
  absltest.main()

if __name__ == '__main__': app.run(main)
