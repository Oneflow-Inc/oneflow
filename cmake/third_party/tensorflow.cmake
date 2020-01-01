include (ExternalProject)

if (WITH_XLA)

list(APPEND TENSORFLOW_BUILD_CMD --define with_xla_support=true)
if (RELEASE_VERSION)
  list(APPEND TENSORFLOW_BUILD_CMD -c opt)
  set(TENSORFLOW_GENFILE_DIR k8-opt)
else()
  list(APPEND TENSORFLOW_BUILD_CMD --copt=-g -c dbg)
  set(TENSORFLOW_GENFILE_DIR k8-dbg)
endif()

set(TF_WITH_CUDA ON)
if (TF_WITH_CUDA)
  set(CUDA_COMPUTE_CAPABILITIES "6.0,6.1")
  if (NOT CUDA_VERSION VERSION_LESS "10.0")
    set(CUDA_COMPUTE_CAPABILITIES "${CUDA_COMPUTE_CAPABILITIES},7.0")
  endif()
  list(APPEND TENSORFLOW_BUILD_CMD --config=cuda)
  list(APPEND TENSORFLOW_BUILD_CMD --action_env TF_NEED_CUDA=1)
  list(APPEND TENSORFLOW_BUILD_CMD --action_env TF_CUDA_COMPUTE_CAPABILITIES=${CUDA_COMPUTE_CAPABILITIES})
endif()

message(STATUS ${TENSORFLOW_BUILD_CMD})

set(TENSORFLOW_PROJECT  tensorflow)
set(TENSORFLOW_GIT_URL  https://github.com/tensorflow/tensorflow.git)
#set(TENSORFLOW_GIT_TAG  master)
set(TENSORFLOW_GIT_TAG  80c04b80ad66bf95aa3f41d72a6bba5e84a99622)
set(TENSORFLOW_SOURCES_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/tensorflow)
set(TENSORFLOW_SRCS_DIR ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)
set(TENSORFLOW_INC_DIR  ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)

set(TENSORFLOW_INSTALL_DIR ${THIRD_PARTY_DIR}/tensorflow)

set(PATCHES_DIR  ${PROJECT_SOURCE_DIR}/oneflow/xrt/patches)
set(TENSORFLOW_JIT_DIR ${TENSORFLOW_SRCS_DIR}/tensorflow/compiler/jit)

set(TENSORFLOW_GEN_DIR ${TENSORFLOW_SRCS_DIR}/bazel-out/${TENSORFLOW_GENFILE_DIR}/genfiles)
set(TENSORFLOW_EXTERNAL_DIR ${TENSORFLOW_SRCS_DIR}/bazel-tensorflow/external)
set(THIRD_ABSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_google_absl)
set(THIRD_PROTOBUF_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_google_protobuf/src)
set(THIRD_BORINGSSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/boringssl/src)
set(THIRD_SNAPPY_DIR ${TENSORFLOW_EXTERNAL_DIR}/snappy)

list(APPEND TENSORFLOW_XLA_INCLUDE_DIR
  ${TENSORFLOW_INC_DIR}
  ${TENSORFLOW_GEN_DIR}
  ${THIRD_ABSL_DIR}
  ${THIRD_PROTOBUF_DIR}
  ${THIRD_BORINGSSL_DIR}
  ${THIRD_SNAPPY_DIR}
)
include_directories(${TENSORFLOW_XLA_INCLUDE_DIR})
list(APPEND TENSORFLOW_XLA_LIBRARIES libtensorflow_framework.so.1)
list(APPEND TENSORFLOW_XLA_LIBRARIES libxla_core.so)
link_directories(${TENSORFLOW_INSTALL_DIR}/lib)

if (THIRD_PARTY)
  ExternalProject_Add(${TENSORFLOW_PROJECT}
    PREFIX ${TENSORFLOW_SOURCES_DIR}
    GIT_REPOSITORY ${TENSORFLOW_GIT_URL}
    GIT_TAG ${TENSORFLOW_GIT_TAG}
    PATCH_COMMAND patch -Np1 < ${PATCHES_DIR}/xla.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd ${TENSORFLOW_SRCS_DIR} &&
                  bazel build ${TENSORFLOW_BUILD_CMD} -j 20 //tensorflow/compiler/jit/xla_lib:libxla_core.so
    INSTALL_COMMAND ""
  )

set(TENSORFLOW_XLA_FRAMEWORK_LIB ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so.1)
set(TENSORFLOW_XLA_CORE_LIB ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow/compiler/jit/xla_lib/libxla_core.so)

add_custom_target(tensorflow_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TENSORFLOW_INSTALL_DIR}/lib
  DEPENDS ${TENSORFLOW_PROJECT})

add_custom_target(tensorflow_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TENSORFLOW_XLA_FRAMEWORK_LIB} ${TENSORFLOW_XLA_CORE_LIB} ${TENSORFLOW_INSTALL_DIR}/lib
  COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${TENSORFLOW_INSTALL_DIR}/lib/libtensorflow_framework.so.1
      ${TENSORFLOW_INSTALL_DIR}/lib/libtensorflow_framework.so
  DEPENDS tensorflow_create_library_dir)
endif(THIRD_PARTY)

endif(WITH_XLA)
