# Oneflow

### 1.1 Linux 

```
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```

### deps

#### for libhdfs3

[libhdfs3](https://github.com/apache/incubator-hawq/blob/master/depends/libhdfs3/README.md)

```bash
    # build
    cmake (2.8+)                    http://www.cmake.org/
    boost (tested on 1.53+)         http://www.boost.org/
    google protobuf                 http://code.google.com/p/protobuf/
    libxml2                         http://www.xmlsoft.org/
    kerberos                        http://web.mit.edu/kerberos/
    libuuid                         http://sourceforge.net/projects/libuuid/
    libgsasl                        http://www.gnu.org/software/gsasl/
```

> Note: boost is not required if c++ compiler is g++ 4.6.0+ or clang++ with stdc++

```bash
    # test
    gtest (tested on 1.7.0)         already integrated in the source code
    gmock (tested on 1.7.0)         already integrated in the source code
```

```bash
    # cov
    gcov (included in gcc distribution)
    lcov (tested on 1.9)            http://ltp.sourceforge.net/coverage/lcov.php
```

### local build

```bash
# for build clean
# remove all files in build but not the `tar.gz`
find ./build -type f -not -name "*.tar.gz" -delete
```

```bash
# you can also use local file instead
cd cmake/third_party

# for macos
ls | xargs sed -i "" "s#http://down.geeek.info/deps/#<local_folder>/#"

# for linux
ls | xargs sed -i "s#http://down.geeek.info/deps/#<local_folder>/#"

# deps in `tar.gz` can be download from `http://down.geeek.info/deps/`

# or make your own use `https://github.com/Oneflow-Inc/deps-archive`
```

### docker

#### binary ignore

```bash
> tree ./docker/bin

./docker/bin
└── clang-format

0 directories, 3 files
```

### submit

```bash
    PATH=/userhome/bin:$PATH CONF_PATH=/userhome/runner/imagenet_30/ LOG_DIR=/userhome/runner/imagenet_30/log submit
```
