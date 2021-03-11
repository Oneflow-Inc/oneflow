import glob
import sys
import os
import re
import importlib

project_source_dir = sys.argv[1]
python_dir = project_source_dir + "/oneflow/python"
output_filepath = sys.argv[2]


def GetImportPath(filepath):
    assert filepath.startswith(python_dir)
    assert filepath.endswith(".py")
    assert len(filepath) > len(python_dir) + len("*.py")
    relative_path = filepath[len(python_dir) : -len(".py")]
    while relative_path[0] == "/":
        relative_path = relative_path[1:]
    relative_path = re.sub(r"/+", ".", relative_path)
    assert re.match(r"^[_\w]+[_\w\d]*(.[_\w]+[_\w\d]*)*$", relative_path)
    return relative_path


def RecursiveFindPythonFile(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            for d in RecursiveFindPythonFile(file_path):
                yield d
        elif file_path.endswith(".py"):
            yield file_path


import_filepaths = []
for py_script in RecursiveFindPythonFile(python_dir):
    file_content = open(py_script, "r", encoding="utf-8").read()
    if re.search(r"@\s?oneflow_export\s?\(", file_content) is not None:
        import_filepaths.append(py_script)

python_scripts = "from __future__ import absolute_import\n"
for filepath in import_filepaths:
    if "onnx" in filepath:
        onnx = importlib.util.find_spec("onnx")
        if onnx is None:
            continue
    python_scripts += "import oneflow.python.%s\n" % GetImportPath(filepath)
open(output_filepath, "w").write(python_scripts)
