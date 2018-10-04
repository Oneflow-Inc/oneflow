#! /bin/bash
lib_name=$1

work_dir="${PWD}/tmp"
target_dir="${PWD}/libs"
mkdir -p ${work_dir} ${target_dir}

function build_libhdfs() {
    echo "====== start building libhdfs ... ======";
    repo_url="https://github.com/apache/hawq/trunk/depends/libhdfs3"
    
    # clone into repo
    cd ${work_dir} \
    && svn export --force ${repo_url} \
    && cd "libhdfs3" \
    && mkdir -p build && cd build \
    && ../bootstrap \
    && make -j \
    && cp ${work_dir}/libhdfs3/build/src/libhdfs3.* ${target_dir}
    echo "====== building libhdfs OK ======";
}

function build_libprotobuf() {
    echo "====== start building libprotobuf ... ======";
    repo_url="https://github.com/mrry/protobuf/trunk/"
    
    #clone into repo
    cd ${work_dir} && svn export --force ${repo_url} \
    && mv trunk protobuf \
    && cd protobuf && ./autogen.sh \
    && ./configure && make -j \
    && cp ${work_dir}/protobuf/src/.libs/* ${work_dir}/protobuf/src/protoc ${target_dir}
    
    echo "====== building libprotobuf OK ======";
}

function build() {
    case $(tr "[:upper:]" "[:lower:]" <<< ${lib_name}) in
        "hdfs")
            build_libhdfs
            ;;
        "protobuf")
            build_libprotobuf
            ;;
        "all")
            build_libhdfs
            build_libprotobuf
            ;;
        *)
            echo "usage: ./build_share_libs.sh <hdfs | protobuf | all>"
            ;;
    esac 
}

build

