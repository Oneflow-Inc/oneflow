import configparser
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "oneflow_src_path", type=str, default=os.getenv("ONEFLOW_SRC_DIR"), required=False
)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(".gitmodules")
for s in config.sections():
    path = config[s]["path"]
    src_path = os.path.join(args.oneflow_src_path, path)
    assert os.path.exists(f"{src_path}/.git"), src_path

    config[s]["url"] = f"file://{src_path}"

with open(".gitmodules", "w") as configfile:
    config.write(configfile)
