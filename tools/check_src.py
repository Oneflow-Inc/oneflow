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
        if python_test_dir in dirpath and "__pycache__" not in dirpath:
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


check_unwanted_test_scripts(
    python_test_dir=os.path.join(src_root, "oneflow/python/test"),
    allowed=["custom_ops", "dataloader", "graph", "models", "modules", "tensor", "util"],
)

check_unwanted_test_scripts(
    python_test_dir=os.path.join(
        src_root, "oneflow/compatible_single_client_python/test"
    ),
    allowed=["models", "ops", "serving", "xrt",],
)
