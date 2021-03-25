import os
import subprocess
import argparse
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("--xla", default=False, action="store_true", required=False)
parser.add_argument("--cuda", type=str, required=False)
parser.add_argument("--src", type=str, required=False)
args = parser.parse_args()

if args.xla:
    assert args.cuda

local_label = ""
version = f"0.3.5"

# set version if release of nightly
assert (
    os.getenv("ONEFLOW_RELEASE_VERSION") != ""
), "ONEFLOW_RELEASE_VERSION should be either None or a valid string"
if os.getenv("ONEFLOW_RELEASE_VERSION"):
    release_version = os.getenv("ONEFLOW_RELEASE_VERSION")
    version = f"{release_version}"
elif os.getenv("ONEFLOW_RELEASE_NIGHTLY"):
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    version += f".dev{date_str}"

# append compute_platform
compute_platform = ""
if args.cuda:
    compute_platform = "".join(args.cuda.split("."))
    assert len(compute_platform) == 3, compute_platform
    compute_platform = "cu" + compute_platform
    if args.xla:
        compute_platform += ".xla"
else:
    compute_platform = "cpu"
assert compute_platform
version += f"+{compute_platform}"

# append git if not release
if not os.getenv("ONEFLOW_RELEASE_VERSION") and not os.getenv(
    "ONEFLOW_RELEASE_NIGHTLY"
):
    try:
        git_hash = (
            subprocess.check_output(
                "git rev-parse --short HEAD", shell=True, cwd=args.src
            )
            .decode()
            .strip()
        )
    except:
        git_hash = "unknown"
    version += f".git.{git_hash}"


dst = os.path.join(args.src, "oneflow/python/version.py")
print(f"-- Generating pip version: {version}, writing to: {dst}")
assert args.src
with open(dst, "w+") as f:
    f.write(f'__version__ = "{version}"')
