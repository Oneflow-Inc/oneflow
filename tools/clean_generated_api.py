import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-root", "--root_path", type=str, required=True)
args = parser.parse_args()


def main():
    for p in glob.glob(os.path.join(args.root_path, "oneflow/*/")):
        if p.endswith("python/") or p.endswith("include/"):
            pass
        else:
            shutil.rmtree(p)


if __name__ == "__main__":
    main()
