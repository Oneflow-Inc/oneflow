cmake_minimum_required(VERSION 3.17.0)
include(ExternalProject)

if(WITH_XLA)

  list(APPEND TENSORFLOW_BUILD_CMD --define with_xla_support=true)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND TENSORFLOW_BUILD_CMD --copt=-g -c dbg)
    set(TENSORFLOW_GENFILE_DIR k8-dbg)
  else()
    list(APPEND TENSORFLOW_BUILD_CMD -c opt)
    set(TENSORFLOW_GENFILE_DIR k8-opt)
  endif()

  list(APPEND TENSORFLOW_BUILD_CMD --config=noaws)
  list(APPEND TENSORFLOW_BUILD_CMD --config=nogcp)
  list(APPEND TENSORFLOW_BUILD_CMD --config=nohdfs)
  list(APPEND TENSORFLOW_BUILD_CMD --config=nonccl)

  set(TF_WITH_CUDA ${BUILD_CUDA})
  if(TF_WITH_CUDA)
    set(CUDA_COMPUTE_CAPABILITIES "6.0,6.1")
    if(NOT CUDA_VERSION VERSION_LESS "10.0")
      set(CUDA_COMPUTE_CAPABILITIES "${CUDA_COMPUTE_CAPABILITIES},7.0")
    endif()
    list(APPEND TENSORFLOW_BUILD_CMD --config=cuda)
    list(APPEND TENSORFLOW_BUILD_CMD --action_env TF_NEED_CUDA=1)
    list(APPEND TENSORFLOW_BUILD_CMD --action_env
         TF_CUDA_COMPUTE_CAPABILITIES=${CUDA_COMPUTE_CAPABILITIES})
  endif()

  message(STATUS "TENSORFLOW_BUILD_CMD: ${TENSORFLOW_BUILD_CMD}")

  set(TENSORFLOW_PROJECT tensorflow)
  set(TENSORFLOW_SOURCES_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow)
  set(TENSORFLOW_SRCS_DIR ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)
  set(TENSORFLOW_INC_DIR ${TENSORFLOW_SOURCES_DIR}/src/tensorflow)

  set(TENSORFLOW_INSTALL_DIR ${THIRD_PARTY_DIR}/tensorflow)

  set(PATCHES_DIR ${PROJECT_SOURCE_DIR}/oneflow/xrt/patches)
  set(TENSORFLOW_JIT_DIR ${TENSORFLOW_SRCS_DIR}/tensorflow/compiler/jit)
  set(TENSORFLOW_GEN_DIR ${TENSORFLOW_SRCS_DIR}/bazel-out/${TENSORFLOW_GENFILE_DIR}/bin)
  set(TENSORFLOW_EXTERNAL_DIR ${TENSORFLOW_SRCS_DIR}/bazel-tensorflow/external)
  set(THIRD_ABSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_google_absl)
  set(THIRD_PROTOBUF_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_google_protobuf/src)
  set(THIRD_BORINGSSL_DIR ${TENSORFLOW_EXTERNAL_DIR}/boringssl/src)
  set(THIRD_SNAPPY_DIR ${TENSORFLOW_EXTERNAL_DIR}/snappy)
  set(THIRD_RE2_DIR ${TENSORFLOW_EXTERNAL_DIR}/com_googlesource_code_re2)

  list(
    APPEND
    TENSORFLOW_XLA_INCLUDE_DIR
    ${TENSORFLOW_INC_DIR}
    ${TENSORFLOW_GEN_DIR}
    ${THIRD_ABSL_DIR}
    ${THIRD_PROTOBUF_DIR}
    ${THIRD_BORINGSSL_DIR}
    ${THIRD_SNAPPY_DIR}
    ${THIRD_RE2_DIR})

  list(
    APPEND
    TENSORFLOW_XLA_INCLUDE_INSTALL_DIR
    "${TENSORFLOW_INSTALL_DIR}/include/tensorflow_inc"
    "${TENSORFLOW_INSTALL_DIR}/include/tensorflow_gen"
    "${TENSORFLOW_INSTALL_DIR}/include/absl"
    "${TENSORFLOW_INSTALL_DIR}/include/protobuf"
    "${TENSORFLOW_INSTALL_DIR}/include/boringssl"
    "${TENSORFLOW_INSTALL_DIR}/include/snappy"
    "${TENSORFLOW_INSTALL_DIR}/include/re2")

  list(APPEND TENSORFLOW_XLA_LIBRARIES -ltensorflow_framework)
  list(APPEND TENSORFLOW_XLA_LIBRARIES -lxla_core)
  link_directories(${TENSORFLOW_INSTALL_DIR}/lib)

  set(XRT_TF_DOWNLOAD_NO_EXTRACT OFF)
  set(XRT_TF_URL
      "https://github.com/Oneflow-Inc/tensorflow/archive/7016a22292a607edc4175d07dae263faad31cd04.zip"
      CACHE STRING "")
  message(STATUS "XRT_TF_URL: ${XRT_TF_URL}")

  if(IS_DIRECTORY ${XRT_TF_URL})
    set(XRT_TF_DOWNLOAD_NO_EXTRACT ON)
  else()
    use_mirror(VARIABLE XRT_TF_URL URL ${XRT_TF_URL})
  endif()

  set(XRT_TF_BAZEL_OUTPUT_DIR ${PROJECT_BINARY_DIR}/xla_tf_bazel_output CACHE STRING "")
  if(THIRD_PARTY)
    ExternalProject_Add(
      ${TENSORFLOW_PROJECT}
      PREFIX ${TENSORFLOW_SOURCES_DIR}
      URL ${XRT_TF_URL}
      DOWNLOAD_NO_EXTRACT ${XRT_TF_DOWNLOAD_NO_EXTRACT}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        cd ${TENSORFLOW_SRCS_DIR} && ${BAZEL_ENV_ARGS} bazel
        --output_user_root=${XRT_TF_BAZEL_OUTPUT_DIR} build ${TENSORFLOW_BUILD_CMD} -j HOST_CPUS
        //tensorflow/compiler/jit/xla_lib:libxla_core.so
      INSTALL_COMMAND "")

    set(TENSORFLOW_XLA_FRAMEWORK_LIB
        ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so.2)
    set(TENSORFLOW_XLA_CORE_LIB
        ${TENSORFLOW_SRCS_DIR}/bazel-bin/tensorflow/compiler/jit/xla_lib/libxla_core.so)

    add_custom_target(
      tensorflow_create_library_dir COMMAND ${CMAKE_COMMAND} -E make_directory
                                            ${TENSORFLOW_INSTALL_DIR}/lib
      DEPENDS ${TENSORFLOW_PROJECT})

    add_custom_target(
      tensorflow_copy_libs_to_destination
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORFLOW_XLA_FRAMEWORK_LIB}
              ${TENSORFLOW_XLA_CORE_LIB} ${TENSORFLOW_INSTALL_DIR}/lib
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              ${TENSORFLOW_INSTALL_DIR}/lib/libtensorflow_framework.so.2
              ${TENSORFLOW_INSTALL_DIR}/lib/libtensorflow_framework.so
      DEPENDS tensorflow_create_library_dir)

    add_custom_target(
      tensorflow_create_include_dir COMMAND ${CMAKE_COMMAND} -E make_directory
                                            ${TENSORFLOW_INSTALL_DIR}/include
      DEPENDS ${TENSORFLOW_PROJECT})

    add_custom_target(tensorflow_symlink_headers DEPENDS tensorflow_create_include_dir)

    foreach(src_dst_pair IN ZIP_LISTS TENSORFLOW_XLA_INCLUDE_DIR TENSORFLOW_XLA_INCLUDE_INSTALL_DIR)
      set(src ${src_dst_pair_0})
      set(dst ${src_dst_pair_1})
      add_custom_command(TARGET tensorflow_symlink_headers COMMAND ${CMAKE_COMMAND} -E
                                                                   create_symlink ${src} ${dst})
    endforeach()

    add_custom_command(
      TARGET tensorflow_symlink_headers
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${GLOG_INCLUDE_DIR}/glog
              ${TENSORFLOW_INSTALL_DIR}/include/tensorflow_inc/tensorflow/core/platform/google)

    add_custom_command(
      TARGET tensorflow_symlink_headers
      COMMAND
        patch --forward
        ${TENSORFLOW_INSTALL_DIR}/include/tensorflow_inc/tensorflow/stream_executor/platform/logging.h
        ${CMAKE_SOURCE_DIR}/cmake/third_party/patches/tensorflow-logging.patch || true)

  endif(THIRD_PARTY)

  include_directories(${TENSORFLOW_XLA_INCLUDE_INSTALL_DIR})

endif(WITH_XLA)
