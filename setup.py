# This File better be run with 'develop' cmd other than 'install'
from __future__ import absolute_import

import os
import re
import sys
import argparse
import shutil
import glob

import setuptools
import setuptools.command.develop
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

import subprocess
from contextlib import contextmanager

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--with_xla",
    type='bool',
    default=False,
    help="Package xla libraries if true, otherwise not."
)
parser.add_argument(
    "--with_tvm",
    type='bool',
    default=False,
    help="Package tvm libraries if true, otherwise not"
)
args, remain_args = parser.parse_known_args()
sys.argv = ['setup.py'] + remain_args


REQUIRED_PACKAGES = [
    'numpy',
    'protobuf',
]


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True

packages = find_packages("build/python_scripts")
package_dir = {
    '':'build/python_scripts',
}

include_files = glob.glob("build/python_scripts/oneflow/include/**/*", recursive=True)
include_files = [os.path.relpath(p, "build/python_scripts/oneflow") for p in include_files]
package_data = {'oneflow': ['_oneflow_internal.so'] + include_files}

if args.with_xla:
    packages += ['oneflow.libs']
    package_dir['oneflow.libs'] = 'third_party/tensorflow/lib'
    package_data['oneflow.libs'] = ['libtensorflow_framework.so.1', 'libxla_core.so']
    # Patchelf >= 0.9 is required.
    oneflow_internal_so = "build/python_scripts/oneflow/_oneflow_internal.so"
    rpath = os.popen("patchelf --print-rpath " + oneflow_internal_so).read()
    command = "patchelf --set-rpath '$ORIGIN/:$ORIGIN/libs/:%s' %s" % \
              (rpath.strip(), oneflow_internal_so)
    if os.system(command) != 0:
        raise Exception("Patchelf set rpath failed. command is: %s" % command)

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

class Develop(setuptools.command.develop.develop):
    def run(self):
        setuptools.command.develop.develop.run(self)

        if (not args.with_tvm):
            return

        ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
        TVM_INSTALL_LIB_DIR = os.path.join(ROOT_DIR, 'third_party', 'tvm', 'lib')
        TVM_SRC_DIR = os.path.join(ROOT_DIR, 'build', 'third_party', 'tvm', 'src', 'tvm')
        TVM_SRC_BUILD_DIR = os.path.join(TVM_SRC_DIR, 'build')

        rm_src_build_dir = ['rm', '-rf', TVM_SRC_BUILD_DIR]
        subprocess.check_call(rm_src_build_dir)

        mkdir_src_build_dir = ['mkdir', TVM_SRC_BUILD_DIR]
        subprocess.check_call(mkdir_src_build_dir)

        # make sure the tvm libs used by oneflow xrt and tvm setup.py are the same one,
        # because there will be two duplicate global static registar vars in two libs
        # which lead to multi registry error, even they are same copied libs.
        tvm_libs = glob.glob(os.path.join(TVM_INSTALL_LIB_DIR, '*'))
        for lib in tvm_libs:
            copy_tvm_files = [
                'ln',
                '-s',
                '{}'.format(lib),
                '{}'.format(os.path.join(TVM_SRC_BUILD_DIR, os.path.basename(lib))),
            ]
            subprocess.check_call(copy_tvm_files)

        # must pass 'develop' other than 'install' to setup.py, cuz 'install' will copy tvm lib to 
        # site-packages directory to invalidate the symbol-link
        with cd(os.path.join(ROOT_DIR, 'build', 'third_party', 'tvm', 'src', 'tvm', 'python')):
            subprocess.check_call("{} setup.py develop".format(sys.executable), shell=True)
        with cd(os.path.join(ROOT_DIR, 'build', 'third_party', 'tvm', 'src', 'tvm', 'topi', 'python')):
            subprocess.check_call("{} setup.py develop".format(sys.executable), shell=True)

cmd_class = {
    'develop': Develop
}

setup(
    name='oneflow',
    version='0.0.1',
    url='https://www.oneflow.org/',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    cmdclass=cmd_class,
    package_dir=package_dir,
    package_data=package_data,
    zip_safe=False,
    distclass=BinaryDistribution,
)
