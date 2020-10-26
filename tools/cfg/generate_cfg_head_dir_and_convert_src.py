import os


SCRIPT_DIR = os.path.split(os.path.realpath(__file__))[0]


def get_hear_dir():
    return SCRIPT_DIR + "/include"


def get_convert_template_script():
    return SCRIPT_DIR + "/template_convert.py"


def get_copy_python_file():
    return SCRIPT_DIR + "/copy_proto_python_file.py"


def get_convert_src_file():
    connector = " "
    src_files = [
        SCRIPT_DIR + "/pybind_module_registry.cpp",
    ]
    return connector.join(src_files)


if __name__ == "__main__":
    print(get_hear_dir())
    print(get_convert_template_script())
    print(get_copy_python_file())
    print(get_convert_src_file())
