#!/usr/bin/env python3
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
from typing import List, Optional
from pathlib import Path


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
    await asyncio.gather(
        handle_stream(process.stdout, l), handle_stream(process.stderr, l),
    )
    await process.wait()
    return process.returncode


def download(build_dir, dry=False) -> Optional[List[str]]:
    urls = [
        "https://github.com/Oneflow-Inc/llvm-project/releases/download/update-err-msg-checker/clang-tidy-15.AppImage"
        if os.getenv("CI")
        else "https://oneflow-static.oss-cn-beijing.aliyuncs.com/bin/clang-tidy/linux-x86_64/clang-tidy-15.AppImage",
        "https://raw.githubusercontent.com/oneflow-inc/llvm-project/maybe/clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py",
    ]
    dst_dir = f"{build_dir}/cache/bin"
    dst = [f"{dst_dir}/clang-tidy", f"{dst_dir}/clang-tidy-diff.py"]
    if dry:
        if os.path.isfile(dst[0]) and os.path.isfile(dst[1]):
            return dst
        else:
            None
    else:
        assert subprocess.call(f"mkdir -p {dst_dir}", shell=True) == 0
        for i, _dst in enumerate(dst):
            assert subprocess.call(f"curl -L {urls[i]} -o {_dst}", shell=True) == 0
            assert subprocess.call(f"chmod +x {_dst}", shell=True) == 0
        return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs clang-tidy on all of the source files."
    )
    parser.add_argument(
        "--build_dir", required=True,
    )
    parser.add_argument(
        "--check-error-msg", action="store_true", default=False,
    )
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    downloaded = download(args.build_dir, dry=True)
    if downloaded is None:
        downloaded = download(args.build_dir)
    assert downloaded is not None
    warnings_as_errors = (
        (Path(__file__).parent / "clang_tidy_warnings_as_errors_on_diff")
        .read_text()
        .strip()
    )
    cmd = f"git diff -U0 master | {downloaded[1]} -clang-tidy-binary {downloaded[0]} -path {args.build_dir} -j $(nproc) -p1 -allow-enabling-alpha-checkers -extra-arg=-Xclang -extra-arg=-analyzer-config -extra-arg=-Xclang -extra-arg=aggressive-binary-operation-simplification=true"
    if args.check_error_msg:
        command = f" cd .. && {cmd} -warnings-as-errors='{warnings_as_errors}' && {cmd} -checks=-*,maybe-need-error-msg -warnings-as-errors=* -skip-line-filter"
    else:
        command = f"cd .. && {cmd} -warnings-as-errors='{warnings_as_errors}'"

    ret_code = loop.run_until_complete(run_command(command))
    exit(ret_code)
