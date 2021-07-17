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
    "--out_dir", type=str, default=".cache/extract_from_oneflow_export/oneflow",
)
args = parser.parse_args()
assert args.out_dir
assert args.out_dir != "~"
assert args.out_dir != "/"
subprocess.check_call(f"rm -rf {args.out_dir}", shell=True)
subprocess.check_call(f"mkdir -p {args.out_dir}", shell=True)


def get_dst_path_(export: str = None):
    splits = export.split(".")
    if len(splits) == 1:
        return "__init__.py"
    else:
        splits = splits[0:-1]
        return f"{os.path.join(*splits)}.py"


def get_dst_path(export: str = None):
    path = get_dst_path_(export=export)
    print([export], path)
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


def is_export_decorator(d):
    return (
        isinstance(d, ast.Call)
        and isinstance(d.func, ast.Name)
        and d.func.id == "oneflow_export"
    )


class DstFile:
    def __init__(self, path):
        self.path = path
        self.imports = set()
        self.segs = []

    def append_seg(self, seg):
        self.segs.append(seg)

    def append_import(self, seg):
        self.imports.add(seg)

    def __str__(self) -> str:
        imports = list(self.imports)
        imports.sort()
        return "\n".join(imports + self.segs)


class DstFileDict:
    state = {}

    @classmethod
    def create_if_absent(cls, path=None):
        if path not in cls.state:
            cls.state[path] = DstFile(path)

    @classmethod
    def append_imports(cls, path=None, imports=None):
        cls.create_if_absent(path=path)
        assert isinstance(imports, list)
        for i in imports:
            cls.state[path].append_import(i)

    @classmethod
    def append_seg(cls, path=None, seg=None):
        cls.create_if_absent(path=path)
        cls.state[path].append_seg(seg)

    @classmethod
    def save(cls):
        for path, f in cls.state.items():
            assert path == f.path
            dir_path = os.path.dirname(path)
            if dir_path:
                subprocess.check_call(f"mkdir -p {dir_path}", shell=True)
            with open(path, "w") as dst_f:
                print("[save]", f.path)
                dst_f.write(str(f))
                dst_f.write("\n")


def handle_export(node=None, export_d=None, imports=None):
    f_src_seg = ast.get_source_segment(txt, node)
    assert len(export_d.args) > 0
    for (i, a) in enumerate(export_d.args):
        if i == 0:
            for d in node.decorator_list:
                if is_export_decorator(d) == False:
                    d_src_seg = ast.get_source_segment(txt, d)
                    DstFileDict.append_seg(
                        path=get_dst_path(export=a.value), seg=f"@{d_src_seg}",
                    )
                    DstFileDict.append_imports(
                        path=get_dst_path(export=a.value), imports=imports,
                    )
            DstFileDict.append_seg(
                path=get_dst_path(export=a.value), seg=f"{f_src_seg}\n",
            )
        else:
            DstFileDict.append_seg(
                path=get_dst_path(export=a.value),
                seg=f"{get_rel_import(exportN=a.value, export0=export_d.args[0].value)}\n",
            )


for (dirpath, dirnames, filenames) in os.walk(args.src_dir):
    if "python/test" in dirpath:
        print("[skip]", dirpath)
        continue
    for src_file in filenames:
        if src_file.endswith(".py"):
            print("[exract]", os.path.join(dirpath, src_file))
            with open(os.path.join(dirpath, src_file), "r") as f:
                txt = f.read()
                module = ast.parse(txt)
                is_exported = False
                # print(ast.dump(parsed))
                imports = []
                for node in module.body:
                    if isinstance(node, (ast.ImportFrom, ast.Import)):
                        import_seg = ast.get_source_segment(txt, node)
                        imports.append(import_seg)
                for node in module.body:
                    is_exported = False
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        for d in node.decorator_list:
                            if is_export_decorator(d):
                                is_exported == True
                    if is_exported:
                        for d in node.decorator_list:
                            handle_export(node=node, export_d=d, imports=imports)
                    else:
                        src_seg = ast.get_source_segment(txt, node)
                        dirpath_without_root = dirpath.split("/")[1::]
                        dirpath_without_root = "/".join(dirpath_without_root)

                        def append_seg(path=None, seg=None):
                            path = os.path.join(path)
                            dir_path = os.path.dirname(path)
                            if dir_path:
                                subprocess.check_call(
                                    f"mkdir -p {dir_path}", shell=True
                                )
                            with open(path, "a") as dst_f:
                                dst_f.write(seg)
                                dst_f.write("\n")

                        append_seg(
                            path=os.path.join(
                                args.out_dir, dirpath_without_root, src_file
                            ),
                            seg=f"{src_seg}\n",
                        )
DstFileDict.save()
import sys

subprocess.check_call(
    f"{sys.executable} -m black . --quiet", shell=True, cwd=args.out_dir
)
