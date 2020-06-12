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
        "--fix",
        default=False,
        action="store_true",
        help="If specified, will re-format the source "
        "code instead of comparing the re-formatted "
        "output, defaults to %(default)s",
    )
    arguments = parser.parse_args()

    if arguments.fix:
        isort_proc = Popen(["isort", "-rc", arguments.source_dir])
        isort_proc.communicate()
        black_proc = Popen(["black", arguments.source_dir])
        black_proc.communicate()
    flack_proc = Popen(["flake8", arguments.source_dir])
    flack_proc.communicate()
    sys.exit(flack_proc.returncode)
