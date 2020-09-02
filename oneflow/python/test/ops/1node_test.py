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
import os

import numpy
import oneflow as flow
from absl import app
from absl.testing import absltest
from absl import flags
import multiprocessing
import unittest

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_boolean("parallel", False, "run test case in parallel on ray")

    # ray-mandatory, ignore them
    flags.DEFINE_string("node-ip-address", None, "")
    flags.DEFINE_string("node-manager-port", None, "")
    flags.DEFINE_string("object-store-name", None, "")
    flags.DEFINE_string("raylet-name", None, "")
    flags.DEFINE_string("redis-address", None, "")
    flags.DEFINE_string("config-list", None, "")
    flags.DEFINE_string("temp-dir", None, "")
    flags.DEFINE_string("redis-password", None, "")


define_flags()


def random_port():
    from socket import socket

    ret = -1
    with socket() as s:
        s.bind(("", 0))
        ret = int(s.getsockname()[1])
        s.close()
    assert ret > -1
    return ret


def main(argv):
    if FLAGS.parallel:
        import ray

        single_gpu = {}
        flow.unittest.register_test_cases(
            scope=single_gpu,
            directory=os.path.dirname(os.path.realpath(__file__)),
            filter_by_num_nodes=lambda x: x == 1,
            filter_by_num_gpus=lambda x: x == 1,
            base_class=absltest.TestCase,
        )
        ray.init()
        futures = []

        num_cpus = 4
        total_num_gpus = len(ray.get_gpu_ids())
        if total_num_gpus > 0:
            num_cpus = multiprocessing.cpu_count() / total_num_gpus
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            num_gpus = 0
        else:
            num_gpus = 1
        for k, v in single_gpu.items():

            @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus, max_retries=0)
            def do_test():
                print("running on ray: {}".format(k))
                flow.env.init()
                flow.env.data_port(random_port())
                flow.env.ctrl_port(random_port())
                runner = unittest.TextTestRunner()
                runner.run(v())

                return 0

            futures.append(do_test.remote())
        ray.get(futures)
    else:
        flow.unittest.register_test_cases(
            scope=globals(),
            directory=os.path.dirname(os.path.realpath(__file__)),
            filter_by_num_nodes=lambda x: x == 1,
            filter_by_num_gpus=lambda x: x >= 1,
            base_class=absltest.TestCase,
        )
        flow.env.init()
        absltest.main()


if __name__ == "__main__":
    app.run(main)
