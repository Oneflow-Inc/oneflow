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
import pathlib
import multiprocessing
import subprocess
import os


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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def check_version(bin):
    out = subprocess.check_output(f"{bin} --version", shell=True).decode()
    expected = "clang-format version 11.0.0"
    return expected == out.strip()


def download(dry=False):
    url = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/bin/clang-format/linux-x86/clang-format-11"
    if os.getenv("CI"):
        url = "https://github.com/Oneflow-Inc/oneflow-fmt/raw/master/clang-format/linux-x86/clang-format-11"
    dst_dir = ".cache"
    dst = f"{dst_dir}/clang-format"
    if dry:
        if os.path.isfile(dst):
            return dst
        else:
            None
    else:
        assert subprocess.call(f"mkdir -p {dst_dir}", shell=True) == 0
        assert subprocess.call(f"curl {url} -o {dst}", shell=True) == 0
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
        "--clang_format_binary",
        required=False,
        help="Path to the clang-format binary",
        default="clang-format",
    )
    parser.add_argument(
        "--source_dir", required=True, help="Root directory of the source code"
    )
    parser.add_argument(
        "--fix",
        default=False,
        action="store_true",
        help="If specified, will re-format the source "
        "code instead of comparing the re-formatted "
        "output, defaults to %(default)s",
    )
    parser.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="If specified, only print errors",
    )
    args = parser.parse_args()
    exts = [".h", ".cc", ".cpp", ".cu", ".cuh"]
    files = filter(
        lambda p: p.suffix in exts and str(p).startswith("oneflow/xrt") == False,
        pathlib.Path(args.source_dir).rglob("*"),
    )
    loop = asyncio.get_event_loop()
    files = [str(f) for f in files]
    clang_fmt_args = "-dry-run --Werror"
    if args.fix:
        clang_fmt_args = "-i"
    results = []
    if check_version(args.clang_format_binary) == False:
        downloaded = download(dry=True)
        if downloaded:
            args.clang_format_binary = downloaded
        else:
            args.clang_format_binary = download()
            assert check_version(args.clang_format_binary)
    for chunk in chunks(files, multiprocessing.cpu_count()):
        promises = [
            run_command(f"{args.clang_format_binary} {clang_fmt_args} {f}")
            for f in chunk
        ]
        chunk_results = loop.run_until_complete(asyncio.gather(*promises))
        results.extend(chunk_results)
    assert len(results) == len(files)
    for (r, f) in zip(results, files):
        if r != 0:
            print("[fail]", f)
    assert sum(results) == 0
