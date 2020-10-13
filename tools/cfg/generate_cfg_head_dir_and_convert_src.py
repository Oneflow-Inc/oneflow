import os


SCRIPT_DIR = os.path.split(os.path.realpath(__file__))[0]


def get_hear_dir():
    return SCRIPT_DIR + "/include"


def get_convert_template_scropt():
    return SCRIPT_DIR + "/template_convert.py"


def get_convert_src_file():
    connector = "\n"
    src_files = [
        SCRIPT_DIR + "/pybind_module_registry.cpp",
    ]
    return connector.join(src_files)


if __name__ == "__main__":
    print(get_hear_dir())
    print(get_convert_template_scropt())
    print(get_convert_src_file())
