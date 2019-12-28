from __future__ import absolute_import

import os
import re
import sys
import argparse
import shutil

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--with_xla",
    type='bool',
    default=False,
    help="Package xla libraries if true, otherwise not."
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
package_data = {'oneflow': ['_oneflow_internal.so']}

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

setup(
    name='oneflow',
    version='0.0.1',
    url='https://www.oneflow.org/',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    zip_safe=False,
    distclass=BinaryDistribution,
)
