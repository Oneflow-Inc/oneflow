import configparser
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--oneflow_src_local_path", type=str, required=False)
parser.add_argument("-r", "--oneflow_src_remote_url", type=str, required=False)
args = parser.parse_args()

assert (
    args.oneflow_src_local_path or args.oneflow_src_remote_url
), "require one of oneflow_src_local_path or oneflow_src_remote_url"
config = configparser.ConfigParser()
config.read(".gitmodules")
for s in config.sections():
    path = config[s]["path"]
    if args.oneflow_src_local_path:
        src_path = os.path.join(args.oneflow_src_local_path, path)
        assert os.path.exists("{}/.git".format(src_path)), src_path
        config[s]["url"] = "file://{}".format(src_path)
    else:
        src_path = os.path.join(args.oneflow_src_remote_url, path)
        config[s]["url"] = src_path

with open(".gitmodules", "w") as configfile:
    config.write(configfile)
