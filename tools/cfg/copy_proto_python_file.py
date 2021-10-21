import os
import argparse
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument(
    "-of_proto_python", "--of_proto_python_dir", type=str, required=True
)
parser.add_argument("-src_proto", "--src_proto_files", type=str, required=True)
parser.add_argument("-dst_dir", "--dst_proto_python_dir", type=str, required=True)

args = parser.parse_args()


def copy_proto_files(src_proto_files, of_proto_python_dir, dst_proto_python_dir):
    for proto_files in src_proto_files:
        src_file = os.path.join(of_proto_python_dir, proto_files[:-6] + "_pb2.py")
        dst_file = os.path.join(dst_proto_python_dir, proto_files[:-6] + "_pb2.py")
        dst_file_dir = os.path.dirname(dst_file)

        if not os.path.exists(dst_file_dir):
            if dst_file_dir:
                os.makedirs(dst_file_dir)
        copyfile(src_file, dst_file)


def main():
    src_proto_files = args.src_proto_files.split(" ")
    # copy _pb2.py files
    copy_proto_files(
        src_proto_files, args.of_proto_python_dir, args.dst_proto_python_dir
    )
    # generate __init__.py files
    sub_dirnames = os.walk(args.dst_proto_python_dir)
    for dirpath, dirnames, _ in sub_dirnames:
        for dirname in dirnames:
            if dirname == "__pycache__":
                continue
            init_file_name = os.path.join(dirpath, dirname, "__init__.py")
            if not os.path.exists(init_file_name):
                init_file = open(init_file_name, "w")
                init_file.close


if __name__ == "__main__":
    main()
