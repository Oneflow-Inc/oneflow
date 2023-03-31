include(ExternalProject)

include(FetchContent)

set(FLATCC_URL https://github.com/dvidelabs/flatcc/archive/refs/tags/v0.6.1.tar.gz)
use_mirror(VARIABLE FLATCC_URL URL ${FLATCC_URL})
message(STATUS "Download flatcc from url: ${FLATCC_URL}")

#FetchContent_Declare(flatcc URL ${FLATCC_URL})
#FetchContent_MakeAvailable(flatcc)
FetchContent_Populate(flatcc URL ${FLATCC_URL} SOURCE_DIR flatcc)

set(FLATCC_ROOT ${CMAKE_CURRENT_BINARY_DIR}/flatcc)
set(FLATCC_SRCS
    "${FLATCC_ROOT}/src/runtime/builder.c"
    "${FLATCC_ROOT}/src/runtime/verifier.c"
    "${FLATCC_ROOT}/src/runtime/emitter.c"
    "${FLATCC_ROOT}/src/runtime/json_parser.c"
    "${FLATCC_ROOT}/src/runtime/json_printer.c"
    "${FLATCC_ROOT}/src/runtime/refmap.c"
    "${FLATCC_ROOT}/config/config.h")
set(FLATCC_INCLUDE_DIR ${FLATCC_ROOT}/include)
add_library(flatcc-runtime STATIC ${FLATCC_SRCS})
target_include_directories(flatcc-runtime SYSTEM PUBLIC ${FLATCC_INCLUDE_DIR})

add_executable(
  flatcc-cli
  "${FLATCC_ROOT}/src/cli/flatcc_cli.c"
  "${FLATCC_ROOT}/external/hash/cmetrohash64.c"
  "${FLATCC_ROOT}/external/hash/str_set.c"
  "${FLATCC_ROOT}/external/hash/ptr_set.c"
  "${FLATCC_ROOT}/src/compiler/hash_tables/symbol_table.c"
  "${FLATCC_ROOT}/src/compiler/hash_tables/scope_table.c"
  "${FLATCC_ROOT}/src/compiler/hash_tables/name_table.c"
  "${FLATCC_ROOT}/src/compiler/hash_tables/schema_table.c"
  "${FLATCC_ROOT}/src/compiler/hash_tables/value_set.c"
  "${FLATCC_ROOT}/src/compiler/fileio.c"
  "${FLATCC_ROOT}/src/compiler/parser.c"
  "${FLATCC_ROOT}/src/compiler/semantics.c"
  "${FLATCC_ROOT}/src/compiler/coerce.c"
  "${FLATCC_ROOT}/src/compiler/codegen_schema.c"
  "${FLATCC_ROOT}/src/compiler/flatcc.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_reader.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_sort.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_builder.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_verifier.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_sorter.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_json_parser.c"
  "${FLATCC_ROOT}/src/compiler/codegen_c_json_printer.c"
  "${FLATCC_ROOT}/src/runtime/builder.c"
  "${FLATCC_ROOT}/src/runtime/emitter.c"
  "${FLATCC_ROOT}/src/runtime/refmap.c")
target_include_directories(flatcc-cli PRIVATE "${FLATCC_ROOT}/external" "${FLATCC_ROOT}/include"
                                              "${FLATCC_ROOT}/config")

#set(FLATCC_EXE ${CMAKE_CURRENT_BINARY_DIR}/flatcc-cli PARENT_SCOPE)
set(FLATCC_EXE ${CMAKE_CURRENT_BINARY_DIR}/flatcc-cli)

function(FLATCC_GENERATE SRCS)
  set(${SRCS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_generated.h")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_generated.h"
      COMMAND ${FLATCC_EXE} ARGS --builder --verifier
              --outfile=${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_generated.h -a ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${FLATCC_EXE}
      COMMENT "Running flatcc compiler on ${FIL}"
      VERBATIM)
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  endforeach()
endfunction()
