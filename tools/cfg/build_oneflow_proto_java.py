import os
import re
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--project_source_dir", type=str, required=True)
parser.add_argument("--proto_files", type=str, required=True)
parser.add_argument("--of_proto_java", type=str, required=True)
parser.add_argument("--protoc_exe", type=str, required=True)
parser.add_argument("--protobuf_jar", type=str, required=True)
args = parser.parse_args()


def copy_append_proto(project_source_dir, proto_files, of_proto_java):
    # copy proto_files to of_proto_java, and append java_package option
    of_proto_java_src = os.path.join(of_proto_java, 'src')
    new_proto_files = []

    for proto_file in proto_files.split(' '):
        proto_file = proto_file[(len(project_source_dir) + 1):]
        package_name = re.sub('/[a-zA-Z_]*.proto', '', proto_file)
        package_name = re.sub('oneflow/core', 'oneflow/proto/core', package_name)
        package_name = re.sub('/', '.', package_name)
        package_name = 'org.' + package_name

        old_file = os.path.join(project_source_dir, proto_file)
        new_file = os.path.join(of_proto_java_src, proto_file)
        new_file_dir = os.path.dirname(new_file)
        if not os.path.exists(new_file_dir):
            if new_file_dir:
                os.makedirs(new_file_dir)
        copyfile(old_file, new_file)
        with open(new_file, 'a+') as f:
            f.write('\noption java_package = "' + package_name + '";\n')
        new_proto_files.append(new_file)

    return new_proto_files


def build_of_proto(protoc_exe, of_proto_java, new_proto_files):
    # call protoc, compile .proto into .java
    of_proto_java_dst = os.path.join(of_proto_java, 'dst')
    of_proto_java_src = os.path.join(of_proto_java, 'src')
    if not os.path.exists(of_proto_java_dst):
        if of_proto_java_dst:
            os.makedirs(of_proto_java_dst)

    command = '{} -I={} --java_out={} {}'.format(
        protoc_exe,
        of_proto_java_src,
        of_proto_java_dst,
        ' '.join(new_proto_files))
    os.system(command)


def build_of_proto_jar(of_proto_java, protobuf_jar):
    target_dir = os.path.join(of_proto_java, 'dst', 'target')
    if not os.path.exists(target_dir):
        if target_dir:
            os.makedirs(target_dir)

    of_proto_java_dst = os.path.join(of_proto_java, 'dst')
    java_files = ''
    for path, subdirs, files in os.walk(of_proto_java_dst):
        for name in files:
            if name.endswith('.java'):
                java_files = java_files + os.path.join(path, name) + ' '
    command = 'javac -source 1.8 -target 1.8 ' + java_files + '-d ' + target_dir + ' -cp ' + protobuf_jar
    print('building proto classes')
    os.system(command)

    oneflow_proto_jar = os.path.join(target_dir, 'oneflow-proto.jar')
    command = 'jar cf ' + oneflow_proto_jar + ' -C ' + target_dir + ' org'
    print('packaging jar')
    os.system(command)


def main():
    project_source_dir = args.project_source_dir
    proto_files = args.proto_files
    of_proto_java = args.of_proto_java
    protoc_exe = args.protoc_exe
    protobuf_jar = args.protobuf_jar

    new_proto_files = copy_append_proto(project_source_dir, proto_files, of_proto_java)
    build_of_proto(protoc_exe, of_proto_java, new_proto_files)
    build_of_proto_jar(of_proto_java, protobuf_jar)


if __name__ == "__main__":
    main()
