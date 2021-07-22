# python3 -m pip install isort autoflake
# requires python3.8 to run
import os
import argparse
import ast
from posixpath import basename, relpath
import subprocess
import multiprocessing
from pathlib import Path, PurePosixPath
from functools import partial
from collections import OrderedDict

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


def print_dump(node):
    print(ast.dump(node))


class SrcFile:
    def __init__(self, spec) -> None:
        is_test = "is_test" in spec and spec["is_test"]
        if is_test:
            print("[skip test]", spec["src"])
        else:
            txt = spec["src"].read_text()
            tree = ast.parse(txt)
            self.node2seg = OrderedDict(
                [(node, ast.get_source_segment(txt, node)) for node in tree.body]
            )
            assert len(self.node2seg.keys()) == len(list(tree.body))
            # self.process_exports()

    def process_exports(self):
        for (node, seg) in self.node2seg.items():
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                print_dump(node)
        # 1. filter exports
        # 2. replace exports with import as


def get_specs_under_python(python_path=None, dst_path=None):
    specs = []
    for p in Path(python_path).rglob("*.py"):
        rel = p.relative_to(python_path)
        dst = PurePosixPath(dst_path).joinpath(rel)
        spec = {"src": p, "dst": dst}
        if rel.parts[0] == "test":
            spec["is_test"] = True
        specs.append(spec)
    return specs


def get_files():
    pool = multiprocessing.Pool()
    srcs = pool.map(
        SrcFile,
        get_specs_under_python(python_path="oneflow/python", dst_path="oneflow")
        + get_specs_under_python(
            python_path="oneflow/compatible_single_client_python",
            dst_path="oneflow/compatible/single/client_python",
        )
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
    return srcs


if __name__ == "__main__":
    subprocess.check_call(f"rm -rf {out_oneflow_dir}", shell=True)
    subprocess.check_call(f"mkdir -p {out_oneflow_dir}", shell=True)
    # step 0: parse and load all segs into memory
    srcs = get_files()
    # step 1: extract all exports
    # step 2: merge files under python/ into generated files
    # step 3: rename all
    # step 4: finalize __all__, if it is imported by another module or wrapped in 'oneflow.export', it should appears in __all__
    # step 5: save file and sort imports and format
