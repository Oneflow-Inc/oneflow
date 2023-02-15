function(SHOW_VARIABLES)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

macro(write_file_if_different file_path content)
  if(EXISTS ${file_path})
    file(READ ${file_path} current_content)
    # NOTE: it seems a cmake bug that "content" in this macro is not
    # treated as a variable
    if(NOT (current_content STREQUAL ${content}))
      file(WRITE ${file_path} ${content})
    endif()
  else()
    file(WRITE ${file_path} ${content})
  endif()
endmacro()

macro(copy_all_files_in_dir source_dir dest_dir target)
  find_program(rsync rsync)
  if(rsync)
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND
        ${rsync}
        # NOTE: the trailing slash of source_dir is needed.
        # Reference: https://stackoverflow.com/a/56627246
        ARGS -a --omit-dir-times --no-perms --no-owner --no-group --inplace ${source_dir}/
        ${dest_dir})
  else()
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                                                           ${source_dir} ${dest_dir})
  endif()
endmacro()

set(_COUNTER 0)
macro(copy_files file_paths source_dir dest_dir target)
  find_program(rsync rsync)
  if(rsync)
    set(CACHE_FILELIST ${PROJECT_BINARY_DIR}/cached_filename_lists/cache_${_COUNTER})
    math(EXPR _COUNTER "${_COUNTER} + 1")
    file(WRITE ${CACHE_FILELIST} "")
    foreach(file ${file_paths})
      file(RELATIVE_PATH rel_path "${source_dir}" ${file})
      file(APPEND ${CACHE_FILELIST} ${rel_path}\n)
    endforeach()
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${rsync} ARGS -a --omit-dir-times --no-perms --no-owner --no-group --inplace
              --files-from=${CACHE_FILELIST} ${source_dir} ${dest_dir})
  else()
    foreach(file ${file_paths})
      file(RELATIVE_PATH rel_path "${source_dir}" ${file})
      add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                                             "${file}" "${dest_dir}/${rel_path}")
    endforeach()
  endif()
endmacro()

function(add_copy_headers_target)
  cmake_parse_arguments(PARSED_ARGS "" "NAME;SRC;DST;INDEX_FILE" "DEPS" ${ARGN})
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "name required")
  endif(NOT PARSED_ARGS_NAME)
  if(NOT PARSED_ARGS_SRC)
    message(FATAL_ERROR "src required")
  endif(NOT PARSED_ARGS_SRC)
  if(NOT PARSED_ARGS_DST)
    message(FATAL_ERROR "dst required")
  endif(NOT PARSED_ARGS_DST)
  add_custom_target(
    "${PARSED_ARGS_NAME}_create_header_dir" COMMAND ${CMAKE_COMMAND} -E make_directory
                                                    "${PARSED_ARGS_DST}"
    DEPENDS ${PARSED_ARGS_DEPS})

  add_custom_target("${PARSED_ARGS_NAME}_copy_headers_to_destination" ALL
                    DEPENDS "${PARSED_ARGS_NAME}_create_header_dir")
  file(GLOB_RECURSE headers "${PARSED_ARGS_SRC}/*.h")
  file(GLOB_RECURSE cuda_headers "${PARSED_ARGS_SRC}/*.cuh")
  file(GLOB_RECURSE hpp_headers "${PARSED_ARGS_SRC}/*.hpp")
  list(APPEND headers ${cuda_headers})
  list(APPEND headers ${hpp_headers})

  foreach(header_file ${headers})
    file(RELATIVE_PATH relative_file_path ${PARSED_ARGS_SRC} ${header_file})
    add_custom_command(
      TARGET "${PARSED_ARGS_NAME}_copy_headers_to_destination" PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file}
              "${PARSED_ARGS_DST}/${relative_file_path}")
  endforeach()

  if(PARSED_ARGS_INDEX_FILE)
    file(STRINGS ${PARSED_ARGS_INDEX_FILE} inventory_headers)
  endif(PARSED_ARGS_INDEX_FILE)
  foreach(header_file ${inventory_headers})
    add_custom_command(
      TARGET "${PARSED_ARGS_NAME}_copy_headers_to_destination" PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PARSED_ARGS_SRC}/${header_file}"
              "${PARSED_ARGS_DST}/${header_file}")
  endforeach()
endfunction()

function(use_mirror)
  set(ALIYUN_URL_PREFIX
      "https://oneflow-static.oss-cn-beijing.aliyuncs.com/third_party_mirror/https/"
      CACHE STRING "URL prefix of Aliyun OSS mirror")
  cmake_parse_arguments(PARSED_ARGS "" "VARIABLE;URL" "" ${ARGN})

  if((NOT PARSED_ARGS_VARIABLE) OR (NOT PARSED_ARGS_URL))
    message(FATAL_ERROR "VARIABLE or URL required")
  endif()

  if(PARSED_ARGS_URL MATCHES "file://")
    set(${PARSED_ARGS_VARIABLE} ${PARSED_ARGS_URL} PARENT_SCOPE)
    return()
  endif()
  if(DEFINED THIRD_PARTY_MIRROR)
    if(THIRD_PARTY_MIRROR STREQUAL "aliyun")
      if(NOT PARSED_ARGS_URL MATCHES "^https://")
        message(FATAL_ERROR "URL should start with 'https://'")
      endif()
      string(REPLACE "https://" ${ALIYUN_URL_PREFIX} MIRRORED_URL ${PARSED_ARGS_URL})
      set(${PARSED_ARGS_VARIABLE} ${MIRRORED_URL} PARENT_SCOPE)
      message(NOTICE "-- fetch ${PARSED_ARGS_VARIABLE} using aliyun mirror ${MIRRORED_URL}")
    elseif(NOT THIRD_PARTY_MIRROR STREQUAL "")
      message(FATAL_ERROR "invalid key for third party mirror")
    endif()
  endif()
endfunction()

macro(set_mirror_url variable url)
  set(${variable} ${url} ${ARGN})
  use_mirror(VARIABLE ${variable} URL ${url})
endmacro()

macro(set_mirror_url_with_hash variable url hash)
  set_mirror_url(${variable} ${url} ${ARGN})
  set(${variable}_HASH ${hash} ${ARGN})
endmacro()

function(check_cxx11_abi OUTPUT_VAR)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E echo "#include <string>\n void test(std::string){}\n int main(){}"
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/temp.cpp)
  try_compile(
    COMPILE_SUCCESS ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/temp.cpp
    COMPILE_DEFINITIONS -D_GLIBCXX_USE_CXX11_ABI=1
    COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/temp)
  if(NOT COMPILE_SUCCESS)
    message(FATAL_ERROR "Detecting cxx11 availability failed. Please report to OneFlow developers.")
  endif()
  execute_process(COMMAND nm ${CMAKE_CURRENT_BINARY_DIR}/temp COMMAND grep -q cxx11
                  RESULT_VARIABLE RET_CODE)
  if(RET_CODE EQUAL 0)
    set(CXX11_ABI_AVAILABLE ON)
  else()
    set(CXX11_ABI_AVAILABLE OFF)
  endif()
  execute_process(COMMAND rm ${CMAKE_CURRENT_BINARY_DIR}/temp ${CMAKE_CURRENT_BINARY_DIR}/temp.cpp)
  set(${OUTPUT_VAR} ${CXX11_ABI_AVAILABLE} PARENT_SCOPE)
endfunction()

include(CheckCXXCompilerFlag)

function(target_try_compile_option target flag)
  # We cannot check for -Wno-foo as this won't throw a warning so we must check for the -Wfoo option directly
  # http://stackoverflow.com/questions/38785168/cc1plus-unrecognized-command-line-option-warning-on-any-other-warning
  string(REGEX REPLACE "^-Wno-" "-W" checkedFlag ${flag})
  string(REGEX REPLACE "[-=]" "_" varName CXX_FLAG${checkedFlag})
  # Avoid double checks. A compiler will not magically support a flag it did not before
  if(NOT DEFINED ${varName}_SUPPORTED)
    check_cxx_compiler_flag(${checkedFlag} ${varName}_SUPPORTED)
  endif()
  if(${varName}_SUPPORTED)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    if(BUILD_CUDA)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_CUDA_COMPILER_ID}" STREQUAL
                                                         "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${flag}>)
      endif()
    endif()
  endif()
endfunction()

function(target_try_compile_options target)
  foreach(flag ${ARGN})
    target_try_compile_option(${target} ${flag})
  endforeach()
endfunction()

function(target_treat_warnings_as_errors target)
  if(TREAT_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    if(BUILD_CUDA)
      # Only pass flags when cuda compiler is Clang because cmake handles -Xcompiler incorrectly
      if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror>)
      endif()
    endif()

    # TODO: remove it while fixing all deprecated call
    target_try_compile_options(${target} -Wno-error=deprecated-declarations)

    # disable unused-* for different compile mode (maybe unused in cpu.cmake, but used in cuda.cmake)
    target_try_compile_options(
      ${target} -Wno-error=unused-const-variable -Wno-error=unused-variable
      -Wno-error=unused-local-typedefs -Wno-error=unused-private-field
      -Wno-error=unused-lambda-capture)

    # there is some strict-overflow warnings in oneflow/user/kernels/ctc_loss_kernel_util.cpp for unknown reason, disable them for now
    target_try_compile_options(${target} -Wno-error=strict-overflow)

    target_try_compile_options(${target} -Wno-error=instantiation-after-specialization)

    # disable for pointer operations of intrusive linked lists
    target_try_compile_options(${target} -Wno-error=array-bounds)

    target_try_compile_options(${target} -Wno-error=comment)

    # disable visibility warnings related to https://github.com/Oneflow-Inc/oneflow/pull/3676.
    target_try_compile_options(${target} -Wno-error=attributes)
  endif()
endfunction()

function(set_compile_options_to_oneflow_target target)
  target_treat_warnings_as_errors(${target})
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
  target_compile_definitions(${target} PRIVATE ONEFLOW_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  # the mangled name between `struct X` and `class X` is different in MSVC ABI, remove it while windows is supported (in MSVC/cl or clang-cl)
  target_try_compile_options(${target} -Wno-covered-switch-default)

  set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

  if(BUILD_CUDA)
    if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
      target_compile_options(
        ${target}
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                -Xcompiler
                -Werror=return-type;
                -Wno-deprecated-gpu-targets;
                -Werror
                cross-execution-space-call;
                -Xcudafe
                --diag_suppress=declared_but_not_referenced;
                >)
    elseif("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
      target_compile_options(
        ${target}
        PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:
          -Werror=return-type;
          # Suppress warning from cub library -- marking as system header seems not working for .cuh files
          -Wno-pass-failed;
          >)
    else()
      message(FATAL_ERROR "Unknown CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")
    endif()
    # remove THRUST_IGNORE_CUB_VERSION_CHECK if starting using bundled cub
    target_compile_definitions(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                                 THRUST_IGNORE_CUB_VERSION_CHECK; >)
  endif()
endfunction()

function(check_variable_defined variable)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "Variable ${variable} is not defined")
  endif()
endfunction()

function(checkDirAndAppendSlash)
  set(singleValues DIR;OUTPUT)
  set(prefix ARG)
  cmake_parse_arguments(PARSE_ARGV 0 ${prefix} "${noValues}" "${singleValues}" "${multiValues}")

  if("${${prefix}_DIR}" STREQUAL "" OR "${${prefix}_DIR}" STREQUAL "/")
    message(FATAL_ERROR "emtpy path found: ${${prefix}_DIR}")
  else()
    set(${${prefix}_OUTPUT} "${${prefix}_DIR}/" PARENT_SCOPE)
  endif()

endfunction()

function(mark_targets_as_system)
  # TODO(daquexian): update this function once https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7308
  # and its following PRs are merged in cmake v3.25.
  foreach(target ${ARGV})
    get_target_property(include_dir ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                               "${include_dir}")
  endforeach()
endfunction()
