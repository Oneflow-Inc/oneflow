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
        help="If specified, will re-format the source, including ISort and Black",
    )

    arguments = parser.parse_args()

    version_cmd = arguments.python_bin + " -m {} --version | grep {} > /dev/null"

    error = False
    for command, version in [
        ("isort", "4.3.21"),
        ("black", "19.10b0"),
    ]:
        command_proc = Popen(version_cmd.format(command, version), shell=True)
        command_proc.communicate()
        if command_proc.returncode:
            print("Linter requires {}=={} !".format(command, version))
            error = True
    if error:
        sys.exit(1)

    if arguments.fix:
        cmd_line = arguments.python_bin + " -m {} " + arguments.source_dir
    else:
        cmd_line = arguments.python_bin + " -m {} " + arguments.source_dir + " --check"

    for py_module in ["isort -rc", "black"]:
        command_proc = Popen(cmd_line.format(py_module), shell=True)
        command_proc.communicate()
        if command_proc.returncode:
            error = True
    sys.exit(1 if error else 0)
