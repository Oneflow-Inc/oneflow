include (ExternalProject)

if (WITH_XLA)

set(TENSORFLOW_PROJECT  tensorflow)
set(TENSORFLOW_GIT_URL  https://github.com/tensorflow/tensorflow.git)
set(TENSORFLOW_GIT_TAG  v1.14.0-rc1)
set(TENSORFLOW_SOURCES_DIR ${THIRD_PARTY_DIR}/tensorflow)
set(TENSORFLOW_SRCS_DIR ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)
set(TENSORFLOW_INC_DIR  ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)

set(XLA_BUILD_PATH  ${PROJECT_SOURCE_DIR}/oneflow/core/compiler/BUILD)
set(TENSORFLOW_DEST_DIR ${TENSORFLOW_SRCS_DIR}/tensorflow/compiler/lib)

set(TENSORFLOW_GEN_DIR ${TENSORFLOW_SRCS_DIR}/bazel-out/k8-opt/genfiles)
set(TENSORFLOW_EXTERNAL_DIR ${TENSORFLOW_SRCS_DIR}/bazel-tensorflow/external)
set(THIRD_ABSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_google_absl)
set(THIRD_PROTOBUF_DIR ${TENSORFLOW_EXTERNAL_DIR}/protobuf_archive/src)
set(THIRD_BORINGSSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/boringssl/src)
set(THIRD_SNAPPY_DIR ${TENSORFLOW_EXTERNAL_DIR}/snappy)


list(APPEND TENSORFLOW_XLA_LIBRARIES libtensorflow_framework.so.1)
list(APPEND TENSORFLOW_XLA_LIBRARIES libxla_computation_client.so)
link_directories(
  ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow
  ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow/compiler/lib
)

list(APPEND TENSORFLOW_XLA_INCLUDE_DIR
  ${TENSORFLOW_INC_DIR}
  ${TENSORFLOW_GEN_DIR}
  ${THIRD_ABSL_DIR}
  ${THIRD_PROTOBUF_DIR}
  ${THIRD_BORINGSSL_DIR}
  ${THIRD_SNAPPY_DIR}
)

if (THIRD_PARTY)
ExternalProject_Add(
  ${TENSORFLOW_PROJECT}
  PREFIX ${TENSORFLOW_SOURCES_DIR}
  GIT_REPOSITORY ${TENSORFLOW_GIT_URL}
  GIT_TAG ${TENSORFLOW_GIT_TAG}
  CONFIGURE_COMMAND mkdir -p ${TENSORFLOW_DEST_DIR} && cp ${XLA_BUILD_PATH} ${TENSORFLOW_DEST_DIR}
  BUILD_COMMAND cd ${TENSORFLOW_SRCS_DIR} && 
                bazel build -c opt --define with_xla_support=true --action_env TF_NEED_CUDA=1 --config=cuda -j 10 //tensorflow/compiler/lib:libxla_computation_client.so
  INSTALL_COMMAND ""
)
endif(THIRD_PARTY)

endif(WITH_XLA)
