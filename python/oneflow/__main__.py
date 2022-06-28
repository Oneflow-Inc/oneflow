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
import os

parser = argparse.ArgumentParser()
parser.add_argument("--doctor", default=False, action="store_true", required=False)
args = parser.parse_args()


def main():
    if args.doctor:
        import oneflow
        import oneflow.sysconfig

        print("path:", oneflow.__path__)
        print("version:", oneflow.__version__)
        print("git_commit:", oneflow.__git_commit__)
        print("cmake_build_type:", oneflow.sysconfig.cmake_build_type())
        print("rdma:", oneflow.sysconfig.with_rdma())
        print("mlir:", oneflow.sysconfig.with_mlir())


if __name__ == "__main__":
    main()
