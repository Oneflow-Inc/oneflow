include (ExternalProject)

if (WITH_XLA)

set(BUILD_DEBUG OFF)
set(TENSORFLOW_BUILD_CMD --define with_xla_support=true)
if (BUILD_DEBUG)
  set(TENSORFLOW_BUILD_CMD --copt=-g -c dbg ${BUILD_CMD})
  set(TENSORFLOW_GENFILE_DIR k8-dbg)
else()
  set(TENSORFLOW_BUILD_CMD -c opt ${BUILD_CMD})
  set(TENSORFLOW_GENFILE_DIR k8-opt)
endif()

set(TF_WITH_CUDA ON)
if (TF_WITH_CUDA)
  set(TENSORFLOW_BUILD_CMD ${TENSORFLOW_BUILD_CMD} --action_env TF_NEED_CUDA=1 --config=cuda)
endif()

set(TENSORFLOW_PROJECT  tensorflow)
set(TENSORFLOW_GIT_URL  https://github.com/tensorflow/tensorflow.git)
set(TENSORFLOW_GIT_TAG  master)
set(TENSORFLOW_SOURCES_DIR ${THIRD_PARTY_DIR}/tensorflow)
set(TENSORFLOW_SRCS_DIR ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)
set(TENSORFLOW_INC_DIR  ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)

set(XLA_BUILD_PATH  ${PROJECT_SOURCE_DIR}/oneflow/xla/xla_lib)
set(TENSORFLOW_DEST_DIR ${TENSORFLOW_SRCS_DIR}/tensorflow/compiler/jit)

set(TENSORFLOW_GEN_DIR ${TENSORFLOW_SRCS_DIR}/bazel-out/${TENSORFLOW_GENFILE_DIR}/genfiles)
set(TENSORFLOW_EXTERNAL_DIR ${TENSORFLOW_SRCS_DIR}/bazel-tensorflow/external)
set(THIRD_ABSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_google_absl)
set(THIRD_PROTOBUF_DIR ${TENSORFLOW_EXTERNAL_DIR}/protobuf_archive/src)
set(THIRD_BORINGSSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/boringssl/src)
set(THIRD_SNAPPY_DIR ${TENSORFLOW_EXTERNAL_DIR}/snappy)


list(APPEND TENSORFLOW_XLA_LIBRARIES libtensorflow_framework.so.1)
list(APPEND TENSORFLOW_XLA_LIBRARIES libxla_core.so)
link_directories(
  ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow
  ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow/compiler/jit/xla_lib
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
  CONFIGURE_COMMAND cp -r ${XLA_BUILD_PATH} ${TENSORFLOW_DEST_DIR}
  BUILD_COMMAND cd ${TENSORFLOW_SRCS_DIR} &&
                bazel build ${TENSORFLOW_BUILD_CMD} -j 20 //tensorflow/compiler/jit/xla_lib:libxla_core.so
  INSTALL_COMMAND ""
)
endif(THIRD_PARTY)

endif(WITH_XLA)
