import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xla", default=False, action="store_true", required=False)
parser.add_argument("--cuda", type=str, required=False)
parser.add_argument("--src", type=str, required=False)
args = parser.parse_args()

if args.xla:
    assert args.cuda

if args.cuda:
    compute_platform = "".join(args.cuda.split("."))
    assert len(compute_platform) == 3, compute_platform
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
            subprocess.check_output("git rev-parse --short HEAD", shell=True, cwd=args.src)
            .decode()
            .strip()
        )
    except:
        git_hash = "unknown"

    version = f"0.3b6+{compute_platform}.git.{git_hash}"

dst = os.path.join(args.src, "oneflow/python/version.py")
print(f"-- Generating pip version: {version} to: {dst}")
assert args.src
with open(dst, "w+") as f:
    f.write(f'__version__ = "{version}"')
