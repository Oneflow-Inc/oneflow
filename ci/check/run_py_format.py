import argparse
import sys
import platform
from subprocess import Popen
import os

if __name__ == "__main__":

    major = platform.sys.version_info.major
    minor = platform.sys.version_info.minor
    if major == 3 and minor < 6:
        print("WARNING: python >= 3.6 required, python source format won't run")
        exit(0)
    parser = argparse.ArgumentParser(
        description="Runs py-format on all of the source files."
        "If --fix is specified enforce format by modifying in place."
    )
    parser.add_argument(
        "--source_dir", required=True, help="Root directory of the source code"
    )
    parser.add_argument(
        "--fix",
        default=False,
        action="store_true",
        help="If specified, will re-format the source",
    )

    arguments = parser.parse_args()
    os.chdir(arguments.source_dir)

    version_cmd = sys.executable + " -m {} --version | grep {} > /dev/null"
    BLACK_VER = "19.10b0"
    if os.system(version_cmd.format("black", BLACK_VER)):
        print(
            'Please install black {black_version}. For instance, run "pip3 install black=={} --user"'.format(
                {"black_version": BLACK_VER}
            )
        )
        sys.exit(1)

    cmd_line = sys.executable + " -m black " + "."
    if arguments.fix == False:
        cmd_line += " --check"
    if os.system(cmd_line):
        sys.exit(1)
