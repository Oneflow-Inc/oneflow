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
import platform
import sys
import subprocess
from contextlib import contextmanager
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.dist import Distribution


# https://github.com/google/or-tools/issues/616
class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--with_tvm",
    action="store_true",
    help="Package tvm libraries if true, otherwise not",
)
parser.add_argument("--build_dir", type=str, default="build")
parser.add_argument("--package_name", type=str, default="oneflow")
args, remain_args = parser.parse_known_args()
sys.argv = ["setup.py"] + remain_args
REQUIRED_PACKAGES = [
    "numpy",
    "protobuf>=3.9.2",
    "tqdm",
    "requests",
]


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


def get_version():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "version", os.path.join("oneflow", "version.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.__version__


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError("Can only cd to absolute path, got: {}".format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


class Develop(develop):
    def run(self):
        develop.run(self)

        if not args.with_tvm:
            return

        ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
        BIULD_DIR = os.path.join(ROOT_DIR, "../build")
        TVM_INSTALL_LIB_DIR = os.path.join(
            BIULD_DIR, "third_party_install", "tvm", "lib"
        )
        TVM_SRC_DIR = os.path.join(
            BIULD_DIR, "third_party", "tvm", "src", "tvm"
        )
        TVM_SRC_BUILD_DIR = os.path.join(TVM_SRC_DIR, "build")

        rm_src_build_dir = ["rm", "-rf", TVM_SRC_BUILD_DIR]
        subprocess.check_call(rm_src_build_dir)

        mkdir_src_build_dir = ["mkdir", TVM_SRC_BUILD_DIR]
        subprocess.check_call(mkdir_src_build_dir)

        # make sure the tvm libs used by oneflow xrt and tvm setup.py are the same one,
        # because there will be two duplicate global static registar vars in two libs
        # which lead to multi registry error, even they are same copied libs.
        tvm_libs = glob.glob(os.path.join(TVM_INSTALL_LIB_DIR, "*"))
        for lib in tvm_libs:
            copy_tvm_files = [
                "ln",
                "-s",
                "{}".format(lib),
                "{}".format(os.path.join(TVM_SRC_BUILD_DIR, os.path.basename(lib))),
            ]
            subprocess.check_call(copy_tvm_files)

        # must pass 'develop' other than 'install' to setup.py, cuz 'install' will copy tvm lib to
        # site-packages directory to invalidate the symbol-link
        with cd(
            os.path.join(
                TVM_SRC_DIR, "python"
            )
        ):
            subprocess.check_call(
                "{} setup.py develop {}".format(sys.executable, " ".join(remain_args)),
                shell=True,
            )
        # with cd(os.path.join(ROOT_DIR, 'build', 'third_party', 'tvm', 'src', 'tvm', 'topi', 'python')):
        #     subprocess.check_call("{} setup.py develop {}".format(sys.executable, ' '.join(remain_args)), shell=True)


cmd_class = {
    "install": InstallPlatlib,
    "develop": Develop,
}

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
    cmdclass=cmd_class,
)
