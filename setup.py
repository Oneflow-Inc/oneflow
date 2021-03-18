from __future__ import absolute_import

import os
import sys
import argparse
import glob
import platform
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install


# https://github.com/google/or-tools/issues/616
class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--with_xla",
    type="bool",
    default=False,
    help="Package xla libraries if true, otherwise not.",
)
parser.add_argument("--build_dir", type=str, default="build")
parser.add_argument("--package_name", type=str, default="oneflow")
args, remain_args = parser.parse_known_args()
sys.argv = ["setup.py"] + remain_args


REQUIRED_PACKAGES = [
    "numpy",
    "protobuf",
    "tqdm",
    "requests",
]

if platform.system() != "Darwin":
    REQUIRED_PACKAGES += ["onnx"]


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


python_scripts_dir = os.path.join(args.build_dir, "python_scripts")
packages = find_packages(python_scripts_dir)
package_dir = {
    "": python_scripts_dir,
}

include_files = glob.glob(
    "{}/python_scripts/oneflow/include/**/*".format(args.build_dir), recursive=True
)
include_files = [
    os.path.relpath(p, "{}/python_scripts/oneflow".format(args.build_dir))
    for p in include_files
]


def get_oneflow_internal_so_path():
    import imp

    fp, pathname, description = imp.find_module(
        "_oneflow_internal", ["{}/python_scripts/oneflow".format(args.build_dir)]
    )
    assert os.path.isfile(pathname)
    return os.path.relpath(pathname, "{}/python_scripts/oneflow".format(args.build_dir))


package_data = {"oneflow": [get_oneflow_internal_so_path()] + include_files}


def get_version():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "version", os.path.join(python_scripts_dir, "oneflow", "python", "version.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.__version__


setup(
    name=args.package_name,
    version=get_version(),
    url="https://www.oneflow.org/",
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
)
