import sys
import os
import argparse
import filecmp
from shutil import copyfile, rmtree
from jinja2 import Environment, FileSystemLoader
import util.proto_reflect_util as proto_reflect_util

parser = argparse.ArgumentParser()
parser.add_argument("-project_build", "--project_build_dir", type=str, required=True)
parser.add_argument("-cfg_workspace", "--cfg_workspace_dir", type=str, required=True)
parser.add_argument("-proto_files", "--proto_file_list", type=str, required=True)
parser.add_argument(
    "-of_cfg_proto_python", "--of_cfg_proto_python_dir", type=str, required=True
)

args = parser.parse_args()

sys.path.insert(0, args.of_cfg_proto_python_dir)

template_dir = os.path.dirname(os.path.abspath(__file__)) + "/template"


def JinjaRender(module, filename, **kwargs):
    j2_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True)
    return j2_env.get_template(filename).render(
        module=module.DESCRIPTOR,
        util=proto_reflect_util.ProtoReflectionUtil(),
        **kwargs
    )


def render_cfg_file(dst_file_path, template_file, module):
    if not os.path.exists(os.path.dirname(dst_file_path)):
        if os.path.dirname(dst_file_path):
            os.makedirs(os.path.dirname(dst_file_path))

    tmp_dst_file = open(dst_file_path, "w")
    tmp_dst_file.write(JinjaRender(module, template_file))
    tmp_dst_file.close()


def convert_hpp(
    dst_hpp_path,
    workspace_dir=args.cfg_workspace_dir,
    project_build_dir=args.project_build_dir,
    module=None,
):
    rel_dst_hpp_path = os.path.relpath(dst_hpp_path, start=args.project_build_dir)
    tmp_hpp_path = workspace_dir + "/" + rel_dst_hpp_path

    render_cfg_file(tmp_hpp_path, "template.cfg.h", module)

    if not os.path.exists(os.path.dirname(dst_hpp_path)):
        if os.path.dirname(dst_hpp_path):
            os.makedirs(os.path.dirname(dst_hpp_path))

    if not os.path.exists(dst_hpp_path) or not filecmp.cmp(tmp_hpp_path, dst_hpp_path):
        copyfile(tmp_hpp_path, dst_hpp_path)


def convert_cpp(
    dst_cpp_path,
    workspace_dir=args.cfg_workspace_dir,
    project_build_dir=args.project_build_dir,
    module=None,
):
    rel_dst_cpp_path = os.path.relpath(dst_cpp_path, start=args.project_build_dir)
    tmp_cpp_path = workspace_dir + "/" + rel_dst_cpp_path

    render_cfg_file(tmp_cpp_path, "template.cfg.cpp", module)

    if not os.path.exists(os.path.dirname(dst_cpp_path)):
        if os.path.dirname(dst_cpp_path):
            os.makedirs(os.path.dirname(dst_cpp_path))

    if not os.path.exists(dst_cpp_path) or not filecmp.cmp(tmp_cpp_path, dst_cpp_path):
        copyfile(tmp_cpp_path, dst_cpp_path)


def convert_pybind(
    dst_pybind_path,
    workspace_dir=args.cfg_workspace_dir,
    project_build_dir=args.project_build_dir,
    module=None,
):
    rel_dst_pybind_path = os.path.relpath(dst_pybind_path, start=args.project_build_dir)
    tmp_pybind_path = workspace_dir + "/" + rel_dst_pybind_path

    render_cfg_file(tmp_pybind_path, "template.cfg.pybind.cpp", module)

    if not os.path.exists(os.path.dirname(dst_pybind_path)):
        if os.path.dirname(dst_pybind_path):
            os.makedirs(os.path.dirname(dst_pybind_path))

    if not os.path.exists(dst_pybind_path) or not filecmp.cmp(
        tmp_pybind_path, dst_pybind_path
    ):
        copyfile(tmp_pybind_path, dst_pybind_path)


def render_template(proto_file_list, generated_file_list):
    for proto_file in proto_file_list:
        rel_proto_file_path, proto_file_name = os.path.split(proto_file)

        proto_py_file_path = args.of_cfg_proto_python_dir + "/" + rel_proto_file_path
        proto_py_file_name = proto_file_name[:-6] + "_pb2"

        sys.path.insert(0, proto_py_file_path)

        proto_module = ""
        for key in sys.modules.keys():
            if key.endswith(proto_py_file_name):
                proto_module = sys.modules[key]
                break

        if not proto_module:
            proto_module = __import__(proto_py_file_name)

        dst_hpp_path = "%s/%s/%s.cfg.h" % (
            args.project_build_dir,
            rel_proto_file_path,
            proto_file_name[:-6],
        )

        dst_cpp_path = "%s/%s/%s.cfg.cpp" % (
            args.project_build_dir,
            rel_proto_file_path,
            proto_file_name[:-6],
        )

        dst_pybind_path = "%s/%s/%s.cfg.pybind.cpp" % (
            args.project_build_dir,
            rel_proto_file_path,
            proto_file_name[:-6],
        )

        convert_hpp(dst_hpp_path, module=proto_module)
        convert_cpp(dst_cpp_path, module=proto_module)
        convert_pybind(dst_pybind_path, module=proto_module)
        generated_file_list.append(dst_hpp_path)
        generated_file_list.append(dst_cpp_path)
        generated_file_list.append(dst_pybind_path)


def main():
    proto_file_list = args.proto_file_list.split(" ")
    # get old generated cfg files
    old_cfg_files = []
    for dirpath, _, filenames in os.walk(args.project_build_dir):
        for filename in filenames:
            abs_file_path = os.path.join(dirpath, filename)
            if abs_file_path.endswith((".cfg.cpp", ".cfg.h", ".cfg.pybind.cpp")):
                old_cfg_files.append(abs_file_path)

    # use generated_file_list save generated cfg files this time
    generated_file_list = []

    render_template(proto_file_list, generated_file_list)

    # old_cfg_files_need_del = old_cfg_files - generated_file_list
    old_cfg_files_need_del = [
        file_name for file_name in old_cfg_files if file_name not in generated_file_list
    ]

    for file_name in old_cfg_files_need_del:
        if os.path.exists(file_name):
            os.remove(file_name)

    rmtree(args.cfg_workspace_dir)


if __name__ == "__main__":
    main()
