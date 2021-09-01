ARG from
FROM ${from}
WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH; \
    cmake -DTHIRD_PARTY=ON -DONEFLOW=OFF -DCMAKE_BUILD_TYPE=Release -DRELEASE_VERSION=ON .. && make -j prepare_oneflow_third_party
