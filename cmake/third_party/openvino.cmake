include (ExternalProject)

if (WITH_OPENVINO)

SET(OPENVINO_GIT_URL https://github.com/opencv/dldt.git)
SET(OPENVINO_GIT_TAG c6f8c23c349f3ef8bacceaf3203f7cc08e6529de) #2020.1

SET(OPENVINO_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/dldt)
SET(OPENVINO_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/dldt/src/openvino)
SET(OPENVINO_INSTALL_DIR ${THIRD_PARTY_DIR}/dldt)

list(APPEND OPENVINO_INCLUDE_DIR ${OPENVINO_INSTALL_DIR}/deployment_tools/inference_engine/include/)
list(APPEND OPENVINO_INCLUDE_DIR ${OPENVINO_INSTALL_DIR}/include/)
list(APPEND OPENVINO_LIBRARIES ${OPENVINO_INSTALL_DIR}/deployment_tools/inference_engine/lib/intel64/libinference_engine.so)
list(APPEND OPENVINO_LIBRARIES ${OPENVINO_INSTALL_DIR}/lib64/libngraph.so)

if (THIRD_PARTY)

  ExternalProject_Add(openvino
    PREFIX ${OPENVINO_DIR}
    GIT_REPOSITORY ${OPENVINO_GIT_URL}
    GIT_TAG ${OPENVINO_GIT_TAG}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:STRING=${OPENVINO_INSTALL_DIR}
    BUILD_COMMAND cd ${OPENVINO_SOURCE_DIR} && git submodule update --init --recursive && mkdir -p build
      && cd build && cmake .. && make -j24)


endif(THIRD_PARTY)
endif(WITH_OPENVINO)
