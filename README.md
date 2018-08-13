# Oneflow

### 1.1 Linux 

```
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```

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
├── hadoop-2.8.1.tar.gz
├── of_submit-1.0-py2-none-any.whl
└── oneflow

0 directories, 3 files
```

