from __future__ import absolute_import

import os
import re
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

REQUIRED_PACKAGES = [
    'numpy',
    'protobuf',
]


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


setup(
    name='oneflow',
    version='0.0.1',
    url='https://www.oneflow.org/',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages("build/python_scripts"),
    package_dir={'':'build/python_scripts'},
    package_data={
        'oneflow': [
            '_oneflow_internal.so',
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
