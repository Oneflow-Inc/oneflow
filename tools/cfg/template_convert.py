import sys
import os
import argparse
from jinja2 import Environment, FileSystemLoader
import util.proto_reflect_util as proto_reflect_util

parser = argparse.ArgumentParser()
parser.add_argument("-project_build", "--project_build_dir", type=str, required=True)
parser.add_argument("-proto_file", "--proto_file_path", type=str, required=True)
parser.add_argument(
    "-of_cfg_proto_python", "--of_cfg_proto_python_dir", type=str, required=True
)
parser.add_argument(
    "--generate_file_type",
    type=str,
    choices=["cfg.cpp", "cfg.pybind.cpp"],
    required=True,
)

args = parser.parse_args()

sys.path.insert(0, args.of_cfg_proto_python_dir)

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template")


def JinjaRender(module, filename, **kwargs):
    j2_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True)
    return j2_env.get_template(filename).render(
        module=module.DESCRIPTOR,
        util=proto_reflect_util.ProtoReflectionUtil(),
        **kwargs
    )


def render_cfg_file(dst_file_path, template_file, module):
    with open(dst_file_path, "w") as dst_file:
        dst_file.write(JinjaRender(module, template_file))


def convert_hpp(dst_hpp_path, module=None):
    if not os.path.exists(os.path.dirname(dst_hpp_path)):
        if os.path.dirname(dst_hpp_path):
            os.makedirs(os.path.dirname(dst_hpp_path))

    render_cfg_file(dst_hpp_path, "template.cfg.h", module)


def convert_cpp(dst_cpp_path, module=None):
    if not os.path.exists(os.path.dirname(dst_cpp_path)):
        if os.path.dirname(dst_cpp_path):
            os.makedirs(os.path.dirname(dst_cpp_path))

    render_cfg_file(dst_cpp_path, "template.cfg.cpp", module)


def convert_pybind(dst_pybind_path, module=None):
    if not os.path.exists(os.path.dirname(dst_pybind_path)):
        if os.path.dirname(dst_pybind_path):
            os.makedirs(os.path.dirname(dst_pybind_path))

    render_cfg_file(dst_pybind_path, "template.cfg.pybind.cpp", module)


def render_template(proto_file):
    rel_proto_file_path, proto_file_name = os.path.split(proto_file)

    proto_py_file_path = os.path.join(args.of_cfg_proto_python_dir, rel_proto_file_path)
    proto_py_file_name = proto_file_name[:-6] + "_pb2"

    sys.path.insert(0, proto_py_file_path)

    proto_module = __import__(proto_py_file_name)

    if args.generate_file_type == "cfg.cpp":
        dst_hpp_path = os.path.join(
            args.project_build_dir, rel_proto_file_path, proto_file_name[:-6] + ".cfg.h"
        )

        dst_cpp_path = os.path.join(
            args.project_build_dir,
            rel_proto_file_path,
            proto_file_name[:-6] + ".cfg.cpp",
        )

        convert_hpp(dst_hpp_path, module=proto_module)
        convert_cpp(dst_cpp_path, module=proto_module)

    elif args.generate_file_type == "cfg.pybind.cpp":
        dst_pybind_path = os.path.join(
            args.project_build_dir,
            rel_proto_file_path,
            proto_file_name[:-6] + ".cfg.pybind.cpp",
        )

        convert_pybind(dst_pybind_path, module=proto_module)
    else:
        raise NotImplementedError


def main():
    render_template(args.proto_file_path)


if __name__ == "__main__":
    main()
