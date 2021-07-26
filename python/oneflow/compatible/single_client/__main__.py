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
parser.add_argument(
    "--start_worker", default=False, action="store_true", required=False
)
parser.add_argument("--env_proto", type=str, required=False)
args = parser.parse_args()


def StartWorker(env_proto):
    import oneflow._oneflow_internal

    oneflow._oneflow_internal.InitEnv(env_proto, False)


def main():
    start_worker = args.start_worker
    if start_worker:
        env_proto = args.env_proto
        assert os.path.isfile(
            env_proto
        ), "env_proto not found, please check your env_proto path: {}".format(env_proto)
        with open(env_proto, "rb") as f:
            StartWorker(f.read())


if __name__ == "__main__":
    main()
