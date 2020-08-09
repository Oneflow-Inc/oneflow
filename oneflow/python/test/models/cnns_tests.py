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
import imp
import os
import sys

import numpy
import oneflow
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("python_bin", "python3", "python binary program name or filepath.")
flags.DEFINE_boolean(
    "enable_auto_mixed_precision",
    False,
    "automatically change float net to mixed precision net",
)


class TestNetMixin:
    """
    Base Tester
  """

    def setUp(self):
        self.net = ""
        self.tf_loss_dir = ""
        self.of_loss_dir = ""
        self.num_iter = 10
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            self.num_iter = 3
        self.set_params()
        oneflow.clear_default_session()

    def set_params(self):
        pass

    def assert_tolerance_4_mixed_precision(self):
        raise AssertionError

    def run_net(self, num_gpu_per_node, num_node=1, node_list=""):
        net_modudle = _Import(self.net)
        spec = net_modudle.DLNetSpec(FLAGS.enable_auto_mixed_precision)
        spec.num_nodes = num_node
        spec.gpu_num_per_node = num_gpu_per_node
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            spec.iter_num = 3
        net_modudle.main(spec)
        return
        if num_node > 1:
            os.system(
                "{} {}.py -g {} -m -n {}".format(
                    FLAGS.python_bin, self.net, num_gpu_per_node, node_list
                )
            )
        else:
            os.system(
                "{} {}.py -g {}".format(FLAGS.python_bin, self.net, num_gpu_per_node)
            )

    def load_tf_loss(self):
        tf_loss = numpy.load(os.path.join(self.tf_loss_dir, "1n1c.npy"))
        return tf_loss[0 : self.num_iter]

    def load_of_loss(self, test_type):
        path = os.path.join(self.of_loss_dir, test_type + ".npy")
        if os.path.exists(path):
            of_loss = numpy.load(path)
        else:
            of_loss = numpy.zeros(self.num_iter)
        return of_loss[0 : self.num_iter]

    def print_and_check_result(self, result_name):
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            if self.net == "resnet50":
                print("WARNING: skipping check for resnet50 cpu due to GEMM NaN")
                return
        loss_dict = {}
        loss_dict["tensorflow"] = self.load_tf_loss()
        loss_dict["oneflow"] = self.load_of_loss(result_name)

        print("==".ljust(64, "="))
        print(" ".ljust(2, " ") + self.net + " loss report")
        print("==".ljust(64, "="))
        fmt_str = "{:>6}  {:>12}  {:>12}"
        print(fmt_str.format("iter", "tensorflow", "oneflow-" + result_name))
        for i in range(self.num_iter):
            fmt_str = "{:>6}  {:>12.6f}  {:>12.6f}"
            print(
                fmt_str.format(i, loss_dict["tensorflow"][i], loss_dict["oneflow"][i])
            )
        if FLAGS.enable_auto_mixed_precision:
            tolerance = self.assert_tolerance_4_mixed_precision()
            rtol = tolerance["rtol"]
            atol = tolerance["atol"]
            print(
                "assert tolerance for mixed_precision are: rtol", rtol, ", atol", atol
            )
            self.assertTrue(
                numpy.allclose(
                    loss_dict["tensorflow"], loss_dict["oneflow"], rtol=rtol, atol=atol
                )
            )
        else:
            self.assertTrue(
                numpy.allclose(loss_dict["tensorflow"], loss_dict["oneflow"])
            )


class TestAlexNetMixin(TestNetMixin):
    """
    AlexNet Tester
  """

    def set_params(self):
        self.net = "alexnet"
        self.tf_loss_dir = os.path.join(
            "/dataset/PNGS/cnns_model_for_test/tf_loss", self.net
        )
        self.of_loss_dir = os.path.join("./of_loss", self.net)

    def assert_tolerance_4_mixed_precision(self):
        return {"rtol": 1e-5, "atol": 1e-2}


class TestResNet50Mixin(TestNetMixin):
    """
    AlexNet Tester
  """

    def set_params(self):
        self.net = "resnet50"
        self.tf_loss_dir = os.path.join(
            "/dataset/PNGS/cnns_model_for_test/tf_loss", self.net
        )
        self.of_loss_dir = os.path.join("./of_loss", self.net)

    def assert_tolerance_4_mixed_precision(self):
        return {"rtol": 1e-8, "atol": 1e-5}


class TestVgg16Mixin(TestNetMixin):
    """
    Vgg16 Tester
  """

    def set_params(self):
        self.net = "vgg16"
        self.tf_loss_dir = os.path.join(
            "/dataset/PNGS/cnns_model_for_test/tf_loss", self.net
        )
        self.of_loss_dir = os.path.join("./of_loss", self.net)

    def assert_tolerance_4_mixed_precision(self):
        return {"rtol": 1e-4, "atol": 1e-1}  # big tolerance due to running ci on 1080ti


class TestInceptionV3Mixin(TestNetMixin):
    """
    InceptionV3 Tester
  """

    def set_params(self):
        self.net = "inceptionv3"
        self.tf_loss_dir = os.path.join(
            "/dataset/PNGS/cnns_model_for_test/tf_loss", self.net
        )
        self.of_loss_dir = os.path.join("./of_loss", self.net)

    def assert_tolerance_4_mixed_precision(self):
        return {"rtol": 1e-5, "atol": 1e-2}


def _Import(name, globals=None, locals=None, fromlist=None):
    # Fast path: see if the module has already been imported.
    try:
        return sys.modules[name]
    except KeyError:
        pass

    # If any of the following calls raises an exception,
    # there's a problem we can't handle -- let the caller handle it.

    fp, pathname, description = imp.find_module(name)

    try:
        return imp.load_module(name, fp, pathname, description)
    finally:
        # Since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()
