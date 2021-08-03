import os
import re
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument(
    "-project_root", "--project_source_dir", type=str, required=True
)
parser.add_argument("-src_proto", "--src_proto_files", type=str, required=True)
parser.add_argument("-dst_dir", "--dst_proto_java_dir", type=str, required=True)
parser.add_argument("-protoc", "--protoc_exe", type=str, required=True)
parser.add_argument("-protobuf", "--protobuf_jar", type=str, required=True)
args = parser.parse_args()


def copy_append_proto(src_proto_files, project_source_dir, dst_proto_java_dir):
    src_proto_files = src_proto_files.split(' ')
    dst_files = []
    dst_proto_java_dir = os.path.join(dst_proto_java_dir, 'src')
    for proto_file in src_proto_files:
        proto_file = proto_file[(len(project_source_dir) + 1):]
        package_name = re.sub(r'/[a-zA-Z_]*.proto', '', proto_file)
        package_name = re.sub(r'/', '.', package_name)
        package_name = 'org.' + package_name

        src_file = os.path.join(project_source_dir, proto_file)
        dst_file = os.path.join(dst_proto_java_dir, proto_file)
        dst_file_dir = os.path.dirname(dst_file)

        if not os.path.exists(dst_file_dir):
            if dst_file_dir:
                os.makedirs(dst_file_dir)
        copyfile(src_file, dst_file)
        with open(dst_file, 'a+') as f:
            f.write('\noption java_package = "' + package_name + '";\n')
        dst_files.append(dst_file)
    return dst_files


def build_proto(dst_files, dst_proto_java_dir, protoc_exe):
    dst_proto_java_dir_dst = os.path.join(dst_proto_java_dir, 'dst', 'org')
    dst_proto_java_dir_src = os.path.join(dst_proto_java_dir, 'src')
    if not os.path.exists(dst_proto_java_dir_dst):
        if dst_proto_java_dir_dst:
            os.makedirs(dst_proto_java_dir_dst)

    command = '{} -I={} --java_out={} {}'.format(
        protoc_exe, 
        dst_proto_java_dir_src,
        dst_proto_java_dir_dst,
        ' '.join(dst_files))
    os.system(command)


def build_jar(dst_proto_java_dir, protobuf_jar):
    proto_classes_dir = os.path.join(dst_proto_java_dir, 'dst', 'target')
    if not os.path.exists(proto_classes_dir):
        if proto_classes_dir:
            os.makedirs(proto_classes_dir)
    dst_proto_java_dir = os.path.join(dst_proto_java_dir, 'dst', 'org')
    java_files = ''
    for path, subdirs, files in os.walk(dst_proto_java_dir):
        for name in files:
            if name.endswith('.java'):
                java_files = java_files + os.path.join(path, name) + ' '
    command = 'javac -source 1.8 -target 1.8 ' + java_files + '-d ' + proto_classes_dir + ' -cp ' + protobuf_jar
    print('building proto classes')
    os.system(command)

    print('packaging jar')
    oneflow_proto_jar = os.path.join(proto_classes_dir, 'oneflow-proto.jar')
    command = 'jar cf ' + oneflow_proto_jar + ' -C ' + proto_classes_dir + ' org'
    os.system(command)


def main():
    src_proto_files = args.src_proto_files
    project_source_dir = args.project_source_dir
    dst_proto_java_dir = args.dst_proto_java_dir
    protoc_exe = args.protoc_exe
    protobuf_jar = args.protobuf_jar

    dst_files = copy_append_proto(src_proto_files, project_source_dir, dst_proto_java_dir)
    build_proto(dst_files, dst_proto_java_dir, protoc_exe)
    build_jar(dst_proto_java_dir, protobuf_jar)


if __name__ == "__main__":
    main()
