# python3 -m pip install isort autoflake
import os
import argparse
import ast
from posixpath import basename, relpath
import subprocess
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_dir", type=str, default="oneflow/python",
)
parser.add_argument(
    "--out_dir", type=str, default="python",
)
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()
assert args.out_dir
assert args.out_dir != "~"
assert args.out_dir != "/"
out_oneflow_dir = os.path.join(args.out_dir, "oneflow")
subprocess.check_call(f"rm -rf {args.out_dir}", shell=True)
subprocess.check_call(f"mkdir -p {out_oneflow_dir}", shell=True)


def get_dst_path_(export: str = None):
    splits = export.split(".")
    if len(splits) == 1:
        return "__init__.py"
    else:
        splits = splits[0:-1]
        return f"{os.path.join(*splits)}.py"


def get_dst_path(export: str = None):
    path = get_dst_path_(export=export)
    return os.path.join(out_oneflow_dir, path)


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


def is_experimental_decorator(d):
    return isinstance(d, ast.Name) and d.id == "experimental_api"


def is_stable_decorator(d):
    return isinstance(d, ast.Name) and d.id == "stable_api"


class DstFile:
    def __init__(self, path):
        self.path = path
        self.imports = set()
        self.segs = []

    def append_seg(self, seg):
        self.segs.append(seg)

    def prepend_segs(self, segs):
        assert isinstance(segs, list)
        self.segs = segs + self.segs

    def append_import(self, seg):
        self.imports.add(seg)

    def __str__(self) -> str:
        return "\n".join(self.segs)

    def merge_file(self, path=None):
        with open(path, "r") as f:
            txt = f.read()
            module = ast.parse(txt)
            segs = []
            pool = multiprocessing.Pool()

            results = pool.starmap(
                ast.get_source_segment, [(txt, node) for node in module.body]
            )
            pool.close()
            for seg in results:
                if (
                    "Copyright 2020 The OneFlow Authors. All rights reserved."
                    not in seg
                ) and "oneflow.python_gen" not in seg:
                    segs.append(seg)
            self.prepend_segs(segs)


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
        dst_file = cls.state[path]
        for i in imports:
            if i not in dst_file.imports and "__future__" not in i:
                dst_file.append_import(i)
                dst_file.append_seg(i + "\n")

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

    @classmethod
    def merge(cls, from_path=None, to_path=None):
        assert to_path in cls.state
        cls.state[to_path].merge_file(path=from_path)


def get_decorator_segs(node=None):
    segs = []
    for d in node.decorator_list:
        if is_export_decorator(d) == False and is_experimental_decorator(d) == False:
            d_src_seg = ast.get_source_segment(txt, d)
            segs.append(f"@{d_src_seg}")
    return segs


def handle_export(node=None, export_d=None, imports=None, txt=None):
    f_src_seg = ast.get_source_segment(txt, node)
    assert len(export_d.args) > 0, str(ast.dump(export_d))
    for (i, a) in enumerate(export_d.args):
        if i == 0:
            DstFileDict.append_imports(
                path=get_dst_path(export=a.value), imports=imports,
            )
            # TODO: check if name of function identical to one of the exported funcs
            for seg in get_decorator_segs(node):
                DstFileDict.append_seg(
                    path=get_dst_path(export=a.value), seg=f"{seg}\n",
                )
            DstFileDict.append_seg(
                path=get_dst_path(export=a.value), seg=f"{f_src_seg}\n",
            )
        else:
            DstFileDict.append_seg(
                path=get_dst_path(export=a.value),
                seg=f"{get_rel_import(exportN=a.value, export0=export_d.args[0].value)}\n",
            )


def parse_from_file(path):
    with open(path, "r") as f:
        txt = f.read()
        print("[parse]", path)
        return (path, txt, ast.parse(txt))


def get_srcs():
    files_to_extract = []
    for (dirpath, dirnames, filenames) in os.walk(args.src_dir):
        if "python/test" in dirpath or "__pycache__" in dirpath:
            print("[skip]", dirpath)
            continue
        else:
            for src_file in filenames:
                if src_file.endswith(".py") and "__pycache__" not in src_file:
                    files_to_extract.append(os.path.join(dirpath, src_file))
    return files_to_extract


def add_init_py():
    for (dirpath, dirnames, filenames) in os.walk(out_oneflow_dir):
        init_py = os.path.join(dirpath, "__init__.py")
        if os.path.exists(init_py) == False:
            subprocess.check_call(f"touch {init_py}", shell=True)


def handle_copy(src_file=None, node=None):
    dirpath = os.path.dirname(src_file)
    dirpath_without_root = dirpath.split("/")[1::]
    dirpath_without_root = "/".join(dirpath_without_root)

    def append_seg(path=None, node=None):
        path = os.path.join(path)
        dir_path = os.path.dirname(path)
        if dir_path:
            if os.path.exists(dir_path) == False:
                subprocess.check_call(f"mkdir -p {dir_path}", shell=True)
        with open(path, "a") as dst_f:
            seg = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                seg += get_decorator_segs(node)
            seg += [ast.get_source_segment(txt, node)]
            dst_f.write("\n".join(seg))
            dst_f.write("\n")

    basename = os.path.basename(src_file)
    dst = os.path.join(out_oneflow_dir, dirpath_without_root, basename)
    append_seg(
        path=dst, node=node,
    )


def handle_node(node=None, src_file=None):
    is_exported = False
    is_experimental = False
    is_stable = False
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        # TODO: append import inside functions and classes
        for d in node.decorator_list:
            # if oneflow_export only, create #0 and reference it in the rest
            if is_export_decorator(d):
                is_exported = True
                handle_export(
                    node=node,
                    export_d=d,
                    imports=imports,
                    txt=txt,
                    # is_experimental=is_experimental,
                )
            if is_experimental_decorator(d):
                is_experimental = True
                print(ast.dump(d))
            if is_stable_decorator(d):
                is_stable = True
                print(ast.dump(d))
    if is_exported == False:
        handle_copy(src_file=src_file, node=node)


if __name__ == "__main__":

    # step 1: extract all exports
    files_to_extract = get_srcs()

    pool = multiprocessing.Pool()

    results = pool.map(parse_from_file, files_to_extract)
    pool.close()
    for (src_file, txt, module) in results:
        is_exported = False
        # print(ast.dump(parsed))
        imports = []
        for node in module.body:
            if isinstance(node, (ast.ImportFrom, ast.Import)):
                import_seg = ast.get_source_segment(txt, node)
                imports.append(import_seg)
        for node in module.body:
            handle_node(node=node, src_file=src_file)

    # step 2: merge files under python/ into generated files
    DstFileDict.merge(
        from_path="oneflow/init.py",
        to_path=os.path.join(out_oneflow_dir, "__init__.py"),
    )
    # step 3: rename all
    # step 4: finalize __all__, if it is imported by another module or wrapped in 'oneflow.export', it should appears in __all__

    # step 5: save file and sort imports and format
    DstFileDict.save()
    add_init_py()
    import sys

    extra_arg = ""
    if args.verbose == False:
        extra_arg += "--quiet"
    subprocess.check_call(
        f"{sys.executable} -m autoflake --in-place --remove-all-unused-imports --recursive .",
        shell=True,
        cwd=args.out_dir,
    )
    subprocess.check_call(
        f"{sys.executable} -m isort . {extra_arg}", shell=True, cwd=args.out_dir,
    )
    subprocess.check_call(
        f"{sys.executable} -m black . {extra_arg}", shell=True, cwd=args.out_dir,
    )

# QA:
# 1. _oneflow_internal.so: link or build in-place?
# 2. do we need make install? run make install to install it to python directory?
# 3. do we need to merge source in python directory. it is confusing if we already have a top level python directory.
