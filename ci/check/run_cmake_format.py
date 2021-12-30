from subprocess import call
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Runs cmake-format on all of the cmake source files."
    )

    parser.add_argument(
        "--bin", default="cmake-format", help="Path of cmake-format binary"
    )
    parser.add_argument(
        "--fix", default=False, action="store_true", help="Format all sources in place"
    )
    parser.add_argument(
        "--source_dir", default=".", help="Root directory of the source code"
    )

    args = parser.parse_args()

    patterns = [
        "cmake/**/*.cmake",
        "oneflow/**/*.cmake",
        "oneflow/**/CMakeLists.txt",
        "tools/**/*.cmake",
        "tools/**/CMakeLists.txt",
        "CMakeLists.txt",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob(str(Path(args.source_dir) / pattern), recursive=True))

    count = 0
    for file in files:
        cmd = [args.bin, file]
        if args.fix:
            cmd.append("-i")
        else:
            cmd.append("--check")
        count += 0 if call(cmd) == 0 else 1

    total = len(files)
    if args.fix:
        print(f"cmake-format -i done. {total} total")
    else:
        print(f"cmake-format --check done. {count} failed / {total} total")

    exit(0 if count == 0 else 1)
