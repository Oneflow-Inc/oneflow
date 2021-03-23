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
from __future__ import absolute_import

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start_worker", type=str, required=True)
parser.add_argument("--env_proto", type=str, required=True)

args = parser.parse_args()


def import_secondary_module(name, path):
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def import_oneflow_internal2():
    import oneflow
    import os
    from os.path import dirname
    import imp

    fp, pathname, description = imp.find_module(
        "_oneflow_internal", [dirname(__file__)]
    )
    assert os.path.isfile(pathname)
    return import_secondary_module("oneflow_api", pathname)


oneflow_api = import_oneflow_internal2()


def StartWorker(env_proto):
    oneflow_api.InitEnv(env_proto)


def main():
    start_worker = args.start_worker
    if start_worker == "1" or start_worker == "y" or start_worker == "yes":
        env_proto = args.env_proto
        assert os.path.isfile(
            env_proto
        ), "env_proto not found, please check your env_proto path: {}".format(env_proto)
        with open(env_proto, "rb") as f:
            StartWorker(f.read())


if __name__ == "__main__":
    main()
