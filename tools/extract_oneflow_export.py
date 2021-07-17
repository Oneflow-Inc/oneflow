import os
import argparse
import ast
from posixpath import relpath
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_dir", type=str, default="oneflow/python",
)
parser.add_argument(
    "--out_dir", type=str, default=".cache/extract_from_oneflow_export",
)
args = parser.parse_args()
assert args.out_dir
assert args.out_dir != "~"
assert args.out_dir != "/"
subprocess.check_call(f"rm -rf {args.out_dir}", shell=True)
subprocess.check_call(f"mkdir -p {args.out_dir}", shell=True)


def append_seg(path=None, seg=None):
    path = os.path.join(path)
    dir_path = os.path.dirname(path)
    if dir_path:
        subprocess.check_call(f"mkdir -p {dir_path}", shell=True)
    with open(path, "a") as dst_f:
        dst_f.write(seg)
        dst_f.write("\n")


def get_dst_path_(export: str = None):
    splits = export.split(".")
    if len(splits) == 1:
        return "__init__.py"
    else:
        splits = splits[0:-1]
        return f"{os.path.join(*splits)}.py"


def get_dst_path(export: str = None):
    path = get_dst_path_(export=export)
    return os.path.join(args.out_dir, path)


def get_rel_import(exportN: str = None, export0: str = None):
    item0 = export0.split(".")[-1]
    itemN = exportN.split(".")[-1]
    if export0.split(".")[0:-1] == exportN.split(".")[0:-1]:
        return f"{itemN} == {item0}"
    else:
        relpath = ".".join((["oneflow"] + export0.split("."))[0:-1])
        if item0 == itemN:
            return f"from {relpath} import {item0}"
        else:
            return f"from {relpath} import {item0} as {itemN}"


def handle_export(node=None):
    f_src_seg = ast.get_source_segment(txt, node)
    assert len(d.args) > 0
    for (i, a) in enumerate(d.args):
        if i == 0:
            append_seg(
                path=get_dst_path(export=a.value), seg=f"{f_src_seg}\n",
            )
        else:
            append_seg(
                path=get_dst_path(export=a.value),
                seg=f"{get_rel_import(exportN=a.value, export0=d.args[0].value)}\n",
            )


for (dirpath, dirnames, filenames) in os.walk(args.src_dir):
    if "python/test" in dirpath:
        continue
    for src_file in filenames:
        if src_file.endswith(".py"):
            with open(os.path.join(dirpath, src_file), "r") as f:
                txt = f.read()
                parsed = ast.parse(txt)
                is_exported = False
                for node in ast.walk(parsed):
                    if isinstance(node, ast.FunctionDef) or isinstance(
                        node, ast.ClassDef
                    ):
                        for d in node.decorator_list:
                            if (
                                isinstance(d, ast.Call)
                                and isinstance(d.func, ast.Name)
                                and d.func.id == "oneflow_export"
                            ):
                                is_exported == True
                                handle_export(node=node)
                    if isinstance(node, ast.FunctionDef) or isinstance(
                        node, ast.ClassDef
                    ):
                        if is_exported:
                            d_src_seg = ast.get_source_segment(txt, node)
                            print(d_src_seg)
                            # print(node.name)
