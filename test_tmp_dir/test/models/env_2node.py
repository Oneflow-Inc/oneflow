"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import atexit

import oneflow as flow
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "nodes_list", "192.168.1.15,192.168.1.16", "nodes list seperated by comma"
)
flags.DEFINE_integer("ctrl_port", "9524", "control port")


def Init():
    flow.env.machine(FLAGS.nodes_list.split(","))
    flow.env.ctrl_port(FLAGS.ctrl_port)
    flow.deprecated.init_worker(scp_binary=True, use_uuid=True)
    atexit.register(flow.deprecated.delete_worker)
