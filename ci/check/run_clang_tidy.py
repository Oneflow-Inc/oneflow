#!/usr/bin/env python2
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import asyncio
import argparse
import subprocess
import os
import platform


def split_and_print(prefix, text):
    lines = text.decode().splitlines(keepends=True)
    prefixed = ""
    for l in lines:
        prefixed += f"{prefix} {l.strip()}"
    if l.strip():
        print(prefixed, flush=True)


async def handle_stream(stream, cb):
    while True:
        line = await stream.readline()
        if line:
            cb(line)
        else:
            break


async def run_command(cmd=None, dry=False, name=None):
    if dry:
        print(f"[dry] {cmd}")
        return 0
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    l = lambda x: split_and_print(f"[{name}]" if name else "", x)
    # l = lambda x: x
    await asyncio.gather(
        handle_stream(process.stdout, l), handle_stream(process.stderr, l),
    )
    await process.wait()
    return process.returncode


def download(dry=False):
    if platform.system() != "Linux":
        raise ValueError("Please install clang format 11.0.0")
    url = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/bin/clang-tidy/linux-x86_64/clang-tidy.AppImage"
    if os.getenv("CI"):
        url = "https://github.com/Oneflow-Inc/llvm-project/releases/download/latest/clang-tidy-489012f-x86_64.AppImage"
    dst_dir = ".cache/bin"
    dst = f"{dst_dir}/clang-tidy"
    if dry:
        if os.path.isfile(dst):
            return dst
        else:
            None
    else:
        assert subprocess.call(f"mkdir -p {dst_dir}", shell=True) == 0
        assert subprocess.call(f"curl -L {url} -o {dst}", shell=True) == 0
        assert subprocess.call(f"chmod +x {dst}", shell=True) == 0
        return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs clang-format on all of the source "
        "files. If --fix is specified enforce format by "
        "modifying in place, otherwise compare the output "
        "with the existing file and output any necessary "
        "changes as a patch in unified diff format"
    )
    parser.add_argument(
        "--clang_tidy_binary",
        required=False,
        help="Path to the clang-tidy binary.",
        default="clang-tidy",
    )
    parser.add_argument(
        "--source_dir", required=True,
    )
    parser.add_argument(
        "--build_dir", required=True,
    )
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    if not os.path.exists(args.clang_tidy_binary):
        downloaded = download(dry=True)
        if downloaded:
            args.clang_tidy_binary = downloaded
        else:
            args.clang_tidy_binary = download()
    assert subprocess.call(f"chmod +x {args.source_dir}/ci/check/clang_tidy_diff.py", shell=True) == 0
    promises = [
        run_command(
            f"cd .. && git diff -U0 master | {args.source_dir}/ci/check/clang_tidy_diff.py -clang-tidy-binary {args.build_dir}/{args.clang_tidy_binary} -path {args.build_dir} -quiet -j $(nproc) -p1"
        )
    ]
    loop.run_until_complete(asyncio.gather(*promises))

