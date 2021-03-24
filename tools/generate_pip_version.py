import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xla", default=False, action="store_true", required=False)
parser.add_argument("--cuda", type=str, required=False)
parser.add_argument("--dst", type=str, required=False)
args = parser.parse_args()

if args.xla:
    assert args.cuda

if args.cuda:
    compute_platform = "".join(args.cuda.split("."))
    assert len(compute_platform) == 3
    compute_platform = "cu" + compute_platform
    if args.xla:
        compute_platform += ".xla"
else:
    compute_platform = "cpu"

local_label = ""
version = ""

if os.getenv("ONEFLOW_RELEASE_VERSION"):
    release_version = os.getenv("ONEFLOW_RELEASE_VERSION")
    version = f"{release_version}+{compute_platform}"
else:
    try:
        git_hash = (
            subprocess.check_output("git rev-parse --short HEAD", shell=True)
            .decode()
            .strip()
        )
    except:
        git_hash = "unknown"

    version = f"0.3b6+{compute_platform}.git.{git_hash}"

print(f"-- Generating pip version: {version}")
assert args.dst
with open(args.dst, "w+") as f:
    f.write(f'__version__ = "{version}"')
