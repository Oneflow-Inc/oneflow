import os
import subprocess
import argparse
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, required=False)
parser.add_argument("--cmake_project_binary_dir", type=str, required=False)
parser.add_argument("--src", type=str, required=False)
parser.add_argument("--out", type=str, required=False)
args = parser.parse_args()

local_label = ""
version = f"0.9.1"

# set version if release of nightly
assert (
    os.getenv("ONEFLOW_RELEASE_VERSION") != ""
), "ONEFLOW_RELEASE_VERSION should be either None or a valid string"
is_release = False
is_nightly = False
if os.getenv("ONEFLOW_RELEASE_VERSION"):
    release_version = os.getenv("ONEFLOW_RELEASE_VERSION")
    version = f"{release_version}"
    is_release = True
elif os.getenv("ONEFLOW_RELEASE_NIGHTLY"):
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    version += f".dev{date_str}"
    is_nightly = True

# append compute_platform
compute_platform = ""
if args.cuda:
    # TODO: use a proper semver lib to handle versions
    splits = args.cuda.split(".")[0:2]
    assert len(splits) == 2
    compute_platform = "".join(splits)
    compute_platform = "cu" + compute_platform
else:
    compute_platform = "cpu"
assert compute_platform
version += f"+{compute_platform}"

try:
    git_hash = (
        subprocess.check_output("git rev-parse --short HEAD", shell=True, cwd=args.src)
        .decode()
        .strip()
    )
except:
    git_hash = "unknown"

# append git if not release
if not os.getenv("ONEFLOW_RELEASE_VERSION") and not os.getenv(
    "ONEFLOW_RELEASE_NIGHTLY"
):
    version += f".git.{git_hash}"


print(f"-- Generating pip version: {version}, writing to: {args.out}")
assert args.out
with open(args.out, "w+") as f:
    f.write(f'__version__ = "{version}"\n')
    f.write(f'__git_commit__ = "{git_hash}"\n')
    if not (is_nightly or is_release):
        f.write(f'__cmake_project_binary_dir__ = "{args.cmake_project_binary_dir}"\n')
