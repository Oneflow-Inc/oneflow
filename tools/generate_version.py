import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xla", default=False, action="store_true", required=False)
parser.add_argument("--cuda", type=str, required=False)
args = parser.parse_args()

if args.xla:
    assert args.cuda

if args.cuda:
    compute_platform = "cu102"
    if args.xla:
        compute_platform += ".xla"
else:
    compute_platform = "cpu"

local_label = ""
version = ""

if os.getenv("ONEFLOW_RELEASE_VERSION"):
    release_version = os.getenv("ONEFLOW_RELEASE_VERSION")
    version = f"{release_version}+{compute_platform}"
elif os.getenv("ONEFLOW_NIGHTLY_VERSION"):
    nightly_version = os.getenv("ONEFLOW_NIGHTLY_VERSION")
    version = f"{nightly_version}+{compute_platform}"
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

print(f"generating pip version: {version}")
