ARG from
FROM ${from}
WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN cmake -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release -DRELEASE_VERSION=ON .. && make -j
