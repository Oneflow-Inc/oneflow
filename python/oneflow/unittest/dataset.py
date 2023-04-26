import os


def dataset_dir(sub_dir=None):
    base_dir = os.getenv("ONEFLOW_TEST_DATASET_DIR")
    if base_dir == None:
        base_dir = "/dataset"
    if sub_dir == None:
        return base_dir
    else:
        return os.path.join(base_dir, sub_dir)
