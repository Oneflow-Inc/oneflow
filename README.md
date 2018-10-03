# Oneflow

### 1.1 Linux 

```
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```

### about shared libs

the **cmake** options **BUILD_THIRD_PARTY** will download shared libs:

```
    libhdfs3.so
    libprotobuf.so
```

and put it into `build/bin/shared`, you should add it to your `LD_LIBRARY_PATH` manually.

for those who want to build your own **libhdfs3.so** nad **libprotobuf.so**, check the docs bellow. 

* [**libhdfs3**](https://github.com/apache/hawq/blob/master/depends/libhdfs3/README.md)

* [**libprotobuf**](https://github.com/protocolbuffers/protobuf)

