ARG from
FROM ${from}
WORKDIR /workspace/build

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH; \
    cmake -G Ninja -DTHIRD_PARTY=OFF .. && ninja ;

## BUILD WHEEL
WORKDIR /workspace
RUN pip3 install wheel
COPY setup.py /workspace/setup.py
RUN python3 setup.py bdist_wheel

FROM centos:7
WORKDIR /workspace
COPY --from=0 /workspace/dist/*.whl .
COPY --from=0 /workspace/build/bin/oneflow_testexe .
