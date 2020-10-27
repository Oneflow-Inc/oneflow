import sys
import os
import argparse
from jinja2 import Environment, FileSystemLoader
import util.proto_reflect_util as proto_reflect_util

parser = argparse.ArgumentParser()
parser.add_argument("-dst_hpp", "--dst_hpp_path", type=str, required=True)
parser.add_argument("-dst_cpp", "--dst_cpp_path", type=str, required=True)
parser.add_argument("-dst_pybind", "--dst_pybind_path", type=str, required=True)
parser.add_argument("-proto_py", "--proto_py_path", type=str, required=True)
parser.add_argument(
    "-of_proto_python", "--of_proto_python_dir", type=str, required=True
)
args = parser.parse_args()

sys.path.insert(0, args.of_proto_python_dir)
sys.path.insert(0, os.path.dirname(args.proto_py_path))

demo = __import__((args.proto_py_path).split("/")[-1])
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/template"


def JinjaRender(module, filename, **kwargs):
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    return j2_env.get_template(filename).render(
        module=module.DESCRIPTOR,
        util=proto_reflect_util.ProtoReflectionUtil(),
        **kwargs
    )


def convert_hpp(dst_hpp_path):
    if not os.path.exists(os.path.dirname(dst_hpp_path)):
        if os.path.dirname(dst_hpp_path):
            os.makedirs(os.path.dirname(dst_hpp_path))
    dst_file = open(dst_hpp_path, "w")
    dst_file.write(JinjaRender(demo, "template.cfg.h"))
    dst_file.close()


def convert_cpp(dst_cpp_path):
    if not os.path.exists(os.path.dirname(dst_cpp_path)):
        if os.path.dirname(dst_cpp_path):
            os.makedirs(os.path.dirname(dst_cpp_path))
    dst_file = open(dst_cpp_path, "w")
    dst_file.write(JinjaRender(demo, "template.cfg.cpp"))
    dst_file.close()


def convert_pybind(dst_pybind_path):
    if not os.path.exists(os.path.dirname(dst_pybind_path)):
        if os.path.dirname(dst_pybind_path):
            os.makedirs(os.path.dirname(dst_pybind_path))
    dst_file = open(dst_pybind_path, "w")
    dst_file.write(JinjaRender(demo, "template.pybind.cpp"))
    dst_file.close()


def main():
    convert_hpp(args.dst_hpp_path)
    convert_cpp(args.dst_cpp_path)
    convert_pybind(args.dst_pybind_path)


if __name__ == "__main__":
    main()
