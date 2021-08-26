import json


def create_one(name=None):
    return {
        "test_suite": name,
        "cuda_version": "N/A",
        "extra_flags": "N/A",
        "os": ["self-hosted", "linux", "build"],
        "allow_fail": False,
        "python_version": "N/A",
    }


def create_conda(name=None):
    return create_one(name=name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only_clang", type=str, required=False)
    args = parser.parse_args()
    if args.only_clang:
        print(json.dumps([create_conda("cpu-clang")]))
