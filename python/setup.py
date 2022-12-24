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

import argparse
import glob
import os
import sys
import numpy as np

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.dist import Distribution


# https://github.com/google/or-tools/issues/616
class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--package_name", type=str, default="oneflow")
args, remain_args = parser.parse_known_args()
sys.argv = ["setup.py"] + remain_args


def get_version():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "version", os.path.join("oneflow", "version.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.__version__


REQUIRED_PACKAGES = [
    f"numpy>={np.__version__}",
    "protobuf>=3.9.2, <4.0",
    "tqdm",
    "requests",
    "pillow",
    "rich",
]

ONEFLOW_VERSION = get_version()
if "cu11" in ONEFLOW_VERSION and "cu112" not in ONEFLOW_VERSION:
    REQUIRED_PACKAGES.append("nvidia-cudnn-cu11")
    REQUIRED_PACKAGES.append("nvidia-cublas-cu11")

# if python version < 3.7.x, than need pip install dataclasses
if sys.version_info.minor < 7:
    REQUIRED_PACKAGES.append("dataclasses")


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


include_files = glob.glob("oneflow/include/**/*", recursive=True)
include_files = [os.path.relpath(p, "oneflow") for p in include_files]
assert len(include_files) > 0, os.path.abspath("oneflow/include")


def get_oneflow_internal_so_path():
    import imp

    fp, pathname, description = imp.find_module("_oneflow_internal", ["oneflow"])
    assert os.path.isfile(pathname)
    return os.path.relpath(pathname, "oneflow")


package_data = {"oneflow": [get_oneflow_internal_so_path()] + include_files}


setup(
    name=args.package_name,
    version=get_version(),
    url="https://www.oneflow.org/",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_dir={"oneflow": "oneflow"},
    package_data=package_data,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
    entry_points={
        "console_scripts": ["oneflow-mock-torch=oneflow.mock_torch.__main__:main"]
    },
)
