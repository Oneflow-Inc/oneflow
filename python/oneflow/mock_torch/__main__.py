"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "mock",
    choices=["enable", "disable"],
    help="enable/disable mocking 'import torch', default is enable",
    nargs="?",
    default="enable",
)
parser.add_argument("--lazy", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

torch_env = Path(__file__).parent


def main():
    if args.mock == "enable":
        print(
            f"export ONEFLOW_MOCK_TORCH_LAZY={args.lazy}; export ONEFLOW_MOCK_TORCH_VERBOSE={args.verbose}; export PYTHONPATH={str(torch_env)}:$PYTHONPATH"
        )
    elif args.mock == "disable" and "PYTHONPATH" in os.environ:
        paths = os.environ["PYTHONPATH"].rstrip(":").split(":")
        paths = [x for x in paths if x != str(torch_env)]
        path = ":".join(paths)
        print(
            f"export PYTHONPATH={path}; unset ONEFLOW_MOCK_TORCH_LAZY; unset ONEFLOW_MOCK_TORCH_VERBOSE"
        )


if __name__ == "__main__":
    main()
