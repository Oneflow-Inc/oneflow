import os
import subprocess
import argparse
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("--xla", default=False, action="store_true", required=False)
parser.add_argument("--cuda", type=str, required=False)
parser.add_argument("--src", type=str, required=False)
parser.add_argument("--out", type=str, required=False)
args = parser.parse_args()

local_label = ""
version = f"0.5.0"

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
else:
    compute_platform = "cpu"
if args.xla:
    compute_platform += ".xla"
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


print(f"-- Generating pip version: {version}, writing to: {args.out}")
assert args.out
with open(args.out, "w+") as f:
    f.write(f'__version__ = "{version}"')
