FROM oneflow-base
WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN cmake -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release -DRELEASE_VERSION=ON .. && make -j

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH; \
    cmake -DTHIRD_PARTY=OFF .. && make -j $(nproc) ;

## BUILD WHEEL
WORKDIR /workspace
RUN pip3 install wheel
COPY setup.py /workspace/setup.py
RUN python3 setup.py bdist_wheel

FROM oneflow-base
WORKDIR /workspace
COPY --from=0 /workspace/dist/*.whl .
COPY --from=0 /workspace/build/bin/oneflow_testexe .
