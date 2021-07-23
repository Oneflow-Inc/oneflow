ARG from
FROM ${from}
WORKDIR /workspace/build

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH; \
    cmake -DTHIRD_PARTY=OFF -DONEFLOW=ON .. && make -j $(nproc) ;

## BUILD WHEEL
WORKDIR /workspace
COPY setup.py /workspace/setup.py
RUN python3 setup.py bdist_wheel

FROM centos:7
WORKDIR /workspace
COPY --from=0 /workspace/dist/*.whl .
COPY --from=0 /workspace/build/bin/oneflow_testexe .
