from absl import flags
import oneflow as flow
import atexit

FLAGS = flags.FLAGS
flags.DEFINE_string('nodes_list', '192.168.1.15,192.168.1.14', 'nodes list seperated by comma')
flags.DEFINE_integer('ctrl_port', '9524', 'control port')

def Init():
  flow.env.machine(FLAGS.nodes_list.split(','))
  flow.env.ctrl_port(FLAGS.ctrl_port)
  flow.deprecated.init_worker(scp_binary=True, use_uuid=True)
  atexit.register(flow.deprecated.delete_worker)
