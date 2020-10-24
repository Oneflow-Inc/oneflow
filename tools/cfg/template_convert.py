import sys
import os
import argparse
import filecmp
from shutil import copyfile
from jinja2 import Environment, FileSystemLoader
import util.proto_reflect_util as proto_reflect_util

parser = argparse.ArgumentParser()
parser.add_argument("-dst_hpp", "--dst_hpp_path", type=str, required=True)
parser.add_argument("-dst_cpp", "--dst_cpp_path", type=str, required=True)
parser.add_argument("-dst_pybind", "--dst_pybind_path", type=str, required=True)
parser.add_argument("-proto_py", "--proto_py_path", type=str, required=True)
parser.add_argument("-project_build", "--project_build_dir", type=str, required=True)
parser.add_argument("-cfg_workspace", "--cfg_workspace_dir", type=str, required=True)
parser.add_argument(
    "-of_cfg_proto_python", "--of_cfg_proto_python_dir", type=str, required=True
)

args = parser.parse_args()

proto_py_file_path, proto_py_file_name = os.path.split(args.proto_py_path)
sys.path.insert(0, args.of_cfg_proto_python_dir)
sys.path.insert(0, proto_py_file_path)

proto_py_module = __import__(proto_py_file_name)
template_dir = os.path.dirname(os.path.abspath(__file__)) + "/template"


def JinjaRender(module, filename, **kwargs):
    j2_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True)
    return j2_env.get_template(filename).render(
        module=module.DESCRIPTOR,
        util=proto_reflect_util.ProtoReflectionUtil(),
        **kwargs
    )


def render_cfg_file(dst_file_path, template_file):
    if not os.path.exists(os.path.dirname(dst_file_path)):
        if os.path.dirname(dst_file_path):
            # use parameter exist_ok to avoid misjudgment under multithreading
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

    tmp_dst_file = open(dst_file_path, "w")
    tmp_dst_file.write(JinjaRender(proto_py_module, template_file))
    tmp_dst_file.close()


def convert_hpp(
    dst_hpp_path,
    workspace_dir=args.cfg_workspace_dir,
    project_build_dir=args.project_build_dir,
):
    rel_dst_hpp_path = os.path.relpath(dst_hpp_path, start=args.project_build_dir)
    tmp_hpp_path = workspace_dir + "/" + rel_dst_hpp_path

    render_cfg_file(tmp_hpp_path, "template.cfg.h")

    if not os.path.exists(os.path.dirname(dst_hpp_path)):
        if os.path.dirname(dst_hpp_path):
            # use parameter exist_ok to avoid misjudgment under multithreading
            os.makedirs(os.path.dirname(dst_hpp_path), exist_ok=True)

    if not os.path.exists(dst_hpp_path) or not filecmp.cmp(tmp_hpp_path, dst_hpp_path):
        copyfile(tmp_hpp_path, dst_hpp_path)


def convert_cpp(
    dst_cpp_path,
    workspace_dir=args.cfg_workspace_dir,
    project_build_dir=args.project_build_dir,
):
    rel_dst_cpp_path = os.path.relpath(dst_cpp_path, start=args.project_build_dir)
    tmp_cpp_path = workspace_dir + "/" + rel_dst_cpp_path

    render_cfg_file(tmp_cpp_path, "template.cfg.cpp")

    if not os.path.exists(os.path.dirname(dst_cpp_path)):
        if os.path.dirname(dst_cpp_path):
            # use parameter exist_ok to avoid misjudgment under multithreading
            os.makedirs(os.path.dirname(dst_cpp_path), exist_ok=True)

    if not os.path.exists(dst_cpp_path) or not filecmp.cmp(tmp_cpp_path, dst_cpp_path):
        copyfile(tmp_cpp_path, dst_cpp_path)


def convert_pybind(
    dst_pybind_path,
    workspace_dir=args.cfg_workspace_dir,
    project_build_dir=args.project_build_dir,
):
    rel_dst_pybind_path = os.path.relpath(dst_pybind_path, start=args.project_build_dir)
    tmp_pybind_path = workspace_dir + "/" + rel_dst_pybind_path

    render_cfg_file(tmp_pybind_path, "template.pybind.cpp")

    if not os.path.exists(os.path.dirname(dst_pybind_path)):
        if os.path.dirname(dst_pybind_path):
            # use parameter exist_ok to avoid misjudgment under multithreading
            os.makedirs(os.path.dirname(dst_pybind_path), exist_ok=True)

    if not os.path.exists(dst_pybind_path) or not filecmp.cmp(
        tmp_pybind_path, dst_pybind_path
    ):
        copyfile(tmp_pybind_path, dst_pybind_path)


def main():
    convert_hpp(args.dst_hpp_path)
    convert_cpp(args.dst_cpp_path)
    convert_pybind(args.dst_pybind_path)


if __name__ == "__main__":
    main()
