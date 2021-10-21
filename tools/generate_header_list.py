import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--src_path", type=str, required=True)
parser.add_argument("-o", "--dst_file", type=str, required=True)
args = parser.parse_args()


def glob_by_pattern(pattern):
    result = []
    for x in glob.glob(os.path.join(args.src_path, pattern), recursive=True):
        result.append(os.path.relpath(x, args.src_path))
    return result


headers = (
    glob_by_pattern("**/*.h")
    + glob_by_pattern("**/*.hpp")
    + glob_by_pattern("**/*.cuh")
    + glob_by_pattern("**/*.proto")
    + glob_by_pattern("**/*.inc")
)
with open(args.dst_file, "w") as f:
    for item in headers:
        f.write("{}\n".format(item))
