import argparse
import sys
from subprocess import Popen

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Runs py-format on all of the source files."
        "If --fix is specified enforce format by modifying in place."
    )
    parser.add_argument(
        "--source_dir", required=True, help="Root directory of the source code"
    )
    parser.add_argument(
        "--python_bin", default="python3", help="Directory of python3 binary program"
    )
    parser.add_argument(
        "--fix",
        default=False,
        action="store_true",
        help="If specified, will re-format the source "
        "code instead of comparing the re-formatted "
        "output, defaults to %(default)s",
    )

    arguments = parser.parse_args()

    version_cmd = arguments.python_bin + " -m {} --version | grep {}  > /dev/null"

    error = False
    for command, version in [
        ("isort", "4.3.212"),
        ("black", "19.10b02"),
        ("flake8", "3.8.22"),
    ]:
        command_proc = Popen(version_cmd.format(command, version), shell=True)
        command_proc.communicate()
        if command_proc.returncode:
            print("Linter requires {}=={} !".format(command, version))
            error = True

    if error:
        sys.exit(1)

    if arguments.fix:
        isort_proc = Popen(
            [arguments.python_bin, "-m", "isort", "-rc", arguments.source_dir]
        )
        isort_proc.communicate()
        black_proc = Popen([arguments.python_bin, "-m", "black", arguments.source_dir])
        black_proc.communicate()
    flack_proc = Popen([arguments.python_bin, "-m", "flake8", arguments.source_dir])
    flack_proc.communicate()
    sys.exit(flack_proc.returncode)
