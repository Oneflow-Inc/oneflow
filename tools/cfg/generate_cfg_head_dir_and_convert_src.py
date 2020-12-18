import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-message_type",
    "--get_message_type",
    type=str,
    choices=[
        "cfg_include_dir",
        "template_convert_python_script",
        "copy_pyproto_python_script",
        "pybind_registry_cc",
        "template_files",
    ],
    required=True,
)
args = parser.parse_args()

SCRIPT_DIR = os.path.split(os.path.realpath(__file__))[0]


def get_hear_dir():
    return os.path.join(SCRIPT_DIR, "include")


def get_convert_template_script():
    return os.path.join(SCRIPT_DIR, "template_convert.py")


def get_copy_python_file():
    return os.path.join(SCRIPT_DIR, "copy_proto_python_file.py")


def get_convert_src_file():
    # use ';' to produce a list of cmake
    connector = ";"
    src_files = [
        os.path.join(SCRIPT_DIR, "pybind_module_registry.cpp"),
    ]
    return connector.join(src_files)


def get_template_files():
    # use ';' to produce a list of cmake
    connector = ";"
    src_files = [
        os.path.join(SCRIPT_DIR, "template", "template.cfg.h"),
        os.path.join(SCRIPT_DIR, "template", "template.cfg.cpp"),
        os.path.join(SCRIPT_DIR, "template", "template.cfg.pybind.cpp"),
        os.path.join(SCRIPT_DIR, "util", "proto_reflect_util.py"),
    ]
    return connector.join(src_files)


if __name__ == "__main__":
    message_type = args.get_message_type
    if message_type == "cfg_include_dir":
        print(get_hear_dir(), end="")
    elif message_type == "template_convert_python_script":
        print(get_convert_template_script(), end="")
    elif message_type == "copy_pyproto_python_script":
        print(get_copy_python_file(), end="")
    elif message_type == "pybind_registry_cc":
        print(get_convert_src_file(), end="")
    elif message_type == "template_files":
        print(get_template_files(), end="")
    else:
        raise NotImplementedError
