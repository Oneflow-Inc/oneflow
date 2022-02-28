from subprocess import call
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

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
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=cpu_count(),
        help="Specifies the number of jobs (commands) to run simultaneously",
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

    def gen_cmd(file):
        cmd = [args.bin, file]
        cmd.append("-i" if args.fix else "--check")
        return cmd

    tp = ThreadPool(args.jobs)
    res = tp.map_async(call, [gen_cmd(file) for file in files])

    tp.close()
    tp.join()

    count = sum(map(lambda x: 0 if x == 0 else 1, res.get()))
    total = len(files)
    if args.fix:
        print(f"cmake-format -i done. {total} total")
    else:
        print(f"cmake-format --check done. {count} failed / {total} total")

    exit(0 if count == 0 else 1)
