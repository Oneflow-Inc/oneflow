# python3 -m pip install isort autoflake
# requires python3.8 to run
import os
import argparse
import ast
from posixpath import basename, relpath
import subprocess
import multiprocessing
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir", type=str, default="python",
)
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()
assert args.out_dir
assert args.out_dir != "~"
assert args.out_dir != "/"
out_oneflow_dir = os.path.join(args.out_dir, "oneflow")


class SrcFile:
    def __init__(self, path: Path = None, destiny=None) -> None:
        self.destiny = destiny

        # pool = multiprocessing.Pool()
        txt = path.read_text()
        module = ast.parse(txt)
        self.node2seg = dict(
            [(node, ast.get_source_segment(txt, node)) for node in module.body]
        )


def get_files():
    pool = multiprocessing.Pool()
    segs = pool.map(
        SrcFile,
        list(Path("oneflow/python").rglob("*.py"))
        + list(Path("oneflow/compatible_single_client_python").rglob("*.py"))
        + [
            {"src": Path("oneflow/init.py"), "dst": "oneflow/__init__.py"},
            {"src": Path("oneflow/__main__.py"), "dst": "oneflow/__main__.py"},
            {
                "src": Path("oneflow/single_client_init.py"),
                "dst": "oneflow/compatible/single_client/__init__.py",
            },
            {
                "src": Path("oneflow/single_client_main.py"),
                "dst": "oneflow/compatible/single_client/__main__.py",
            },
        ],
    )
    pool.close()
    # for path in Path("oneflow/compatible_single_client_python").rglob("*.py"):
    #     SrcFile(path=path)
    # for path in Path("oneflow/python").rglob("*.py"):
    #     print(path.absolute())


if __name__ == "__main__":
    subprocess.check_call(f"rm -rf {out_oneflow_dir}", shell=True)
    subprocess.check_call(f"mkdir -p {out_oneflow_dir}", shell=True)
    # step 0: parse and load all segs into memory
    get_files()
    # step 1: extract all exports
    # step 2: merge files under python/ into generated files
    # step 3: rename all
    # step 4: finalize __all__, if it is imported by another module or wrapped in 'oneflow.export', it should appears in __all__
    # step 5: save file and sort imports and format
