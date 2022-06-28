import os

this_file = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.join(this_file, "..")
src_root = os.path.abspath(src_root)


def check_unwanted_test_scripts(python_test_dir=None, allowed=None):
    python_test_dir = os.path.abspath(python_test_dir)

    allowed_full = [
        os.path.relpath(os.path.join(python_test_dir, a), src_root) for a in allowed
    ]
    for (dirpath, dirnames, filenames) in os.walk(src_root):
        if (
            dirpath.startswith(os.path.abspath(python_test_dir) + os.sep)
            and "__pycache__" not in dirpath
        ):
            rel_to_python_test = os.path.relpath(dirpath, python_test_dir)
            rel_to_src_root = os.path.relpath(dirpath, src_root)
            print(f"checking: {rel_to_src_root}")
            if (
                rel_to_python_test not in allowed
                and rel_to_python_test != "."
                and "custom_ops" not in rel_to_python_test
            ):
                if filenames == []:
                    raise ValueError(f"delete this directory: {rel_to_src_root}")
                else:
                    filenames_full = [
                        os.path.relpath(os.path.join(dirpath, a), src_root)
                        for a in filenames
                    ]
                    raise ValueError(
                        f"""move these files:
    {filenames_full}
    inside one of these directories:
    {allowed_full},
    and delete this directory: {rel_to_src_root}"""
                    )


def check_dir_empty(path):
    if os.path.exists(path):
        for dirpath, dirnames, files in os.walk(path):
            if files:
                raise ValueError(dirpath, "must be empty")


check_unwanted_test_scripts(
    python_test_dir=os.path.join(src_root, "python/oneflow/test"),
    allowed=[
        "custom_ops",
        "dataloader",
        "graph",
        "models",
        "modules",
        "tensor",
        "exceptions",
        "expensive",
        "misc",
        "profiler",
    ],
)

check_unwanted_test_scripts(
    python_test_dir=os.path.join(
        src_root, "python/oneflow/compatible/single_client/test"
    ),
    allowed=["models", "ops", "serving",],
)

check_dir_empty(os.path.join(src_root, "oneflow/python"))
check_dir_empty(os.path.join(src_root, "oneflow/compatible_single_client_python"))
