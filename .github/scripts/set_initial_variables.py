import json


def create_one(name=None, allow_fail=None):
    return {
        "test_suite": name,
        "cuda_version": "N/A",
        "extra_flags": "N/A",
        "os": ["self-hosted", "linux", "build"],
        "allow_fail": allow_fail,
        "python_version": "N/A",
    }


def create_conda(name=None):
    return create_one(name=name, allow_fail=False)


def print_github_action_output(name=None, value=None):
    print(f"::set-output name={name}::{value}")


def print_result(build_matrix=None, test_matrix=None, out=None):
    check_include(include_key="test_suite", matrix=build_matrix)
    if test_matrix != {}:
        check_include(include_key="test_suite", matrix=test_matrix)
    assert build_matrix
    assert test_matrix != None
    root = {
        "build_matrix": build_matrix,
        "test_matrix": test_matrix,
    }
    for k, v in root.items():
        print_github_action_output(
            name=k, value=json.dumps(v),
        )
    if out:
        with open(out, "w+") as f:
            f.write(json.dumps(root, indent=4))


def check_include(include_key=None, matrix: dict = None):
    assert include_key in matrix
    in_declare = set(matrix[include_key])
    in_include = set()
    for include_value in matrix["include"]:
        in_include.add(include_value[include_key])
    assert in_declare == in_include, {
        "in_declare": in_declare,
        "in_include": in_include,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", type=lambda x: (str(x).replace(" ", "").split(",")), required=True,
    )
    parser.add_argument("--out", type=str, required=False)
    args = parser.parse_args()
    if "need-clang-only" in args.labels:
        print_result(
            build_matrix={
                "test_suite": ["cpu-clang"],
                "include": [create_conda("cpu-clang")],
            },
            test_matrix={},
            out=args.out,
        )
    else:
        full_build_matrix = {
            "test_suite": ["cuda", "cpu", "xla", "xla_cpu", "cpu-clang"],
            "include": [
                {
                    "test_suite": "cuda",
                    "cuda_version": 10.2,
                    "extra_flags": "--extra_oneflow_cmake_args=-DCUDA_ARCHITECTURES=61 --extra_oneflow_cmake_args=-DRPC_BACKEND=GRPC,LOCAL --extra_oneflow_cmake_args=-DPIP_INDEX_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple",
                    "os": ["self-hosted", "linux", "build"],
                    "allow_fail": False,
                    "python_version": "3.6,3.7",
                },
                {
                    "test_suite": "cpu",
                    "cuda_version": 10.2,
                    "extra_flags": "--extra_oneflow_cmake_args=-DBUILD_SHARED_LIBS=OFF --extra_oneflow_cmake_args=-DRPC_BACKEND=LOCAL --cpu",
                    "os": ["self-hosted", "linux", "build"],
                    "allow_fail": False,
                    "python_version": "3.6,3.7",
                },
                {
                    "test_suite": "xla",
                    "cuda_version": 10.1,
                    "extra_flags": "--extra_oneflow_cmake_args=-DCUDA_ARCHITECTURES=61 --extra_oneflow_cmake_args=-DRPC_BACKEND=GRPC,LOCAL --xla --extra_oneflow_cmake_args=-DPIP_INDEX_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple",
                    "os": ["self-hosted", "linux", "build"],
                    "allow_fail": True,
                    "python_version": 3.6,
                },
                {
                    "test_suite": "xla_cpu",
                    "cuda_version": 10.1,
                    "extra_flags": "--extra_oneflow_cmake_args=-DRPC_BACKEND=GRPC,LOCAL --xla --cpu --extra_oneflow_cmake_args=-DPIP_INDEX_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple",
                    "os": ["self-hosted", "linux", "build"],
                    "allow_fail": True,
                    "python_version": 3.6,
                },
                create_conda("cpu-clang"),
            ],
        }
        full_test_matrix = {
            "test_suite": [
                "cuda",
                "cuda_op",
                "cuda_new_interface",
                "cpu_new_interface",
                "cpu",
                "xla",
                "xla_cpu",
            ],
            "include": [
                {
                    "test_suite": "cuda",
                    "os": ["self-hosted", "linux", "gpu"],
                    "allow_fail": False,
                    "build_env": "build.cuda.env",
                },
                {
                    "test_suite": "cuda_op",
                    "os": ["self-hosted", "linux", "gpu"],
                    "allow_fail": False,
                    "build_env": "build.cuda.env",
                },
                {
                    "test_suite": "cuda_new_interface",
                    "os": ["self-hosted", "linux", "gpu"],
                    "allow_fail": False,
                    "build_env": "build.cuda.env",
                },
                {
                    "test_suite": "cpu",
                    "os": ["self-hosted", "linux", "cpu"],
                    "allow_fail": False,
                    "build_env": "build.cpu.env",
                },
                {
                    "test_suite": "cpu_new_interface",
                    "os": ["self-hosted", "linux", "cpu"],
                    "allow_fail": False,
                    "build_env": "build.cpu.env",
                },
                {
                    "test_suite": "xla",
                    "os": ["self-hosted", "linux", "gpu"],
                    "allow_fail": True,
                    "build_env": "build.xla.env",
                },
                {
                    "test_suite": "xla_cpu",
                    "os": ["self-hosted", "linux", "cpu"],
                    "allow_fail": True,
                    "build_env": "build.xla_cpu.env",
                },
            ],
        }
        print_result(
            build_matrix=full_build_matrix, test_matrix=full_test_matrix, out=args.out,
        )
