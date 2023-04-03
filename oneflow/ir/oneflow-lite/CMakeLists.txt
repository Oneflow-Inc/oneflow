include_directories(${PROJECT_BINARY_DIR}/oneflow-lite)
include_directories(${PROJECT_SOURCE_DIR}/oneflow-lite)
include_directories(${PROJECT_SOURCE_DIR}/oneflow-lite/include)
include_directories(${PROJECT_BINARY_DIR}/oneflow-lite/include)

add_subdirectory(schemas)
add_subdirectory(lib)

set(LLVM_LINK_COMPONENTS Support)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(
  oneflow-lite-compile
  OneFlowLiteCompileMain.cpp
  DEPENDS
  MLIROneFlow
  lite_schemas
  OneFlowLiteConversion
  flatcc-runtime)

set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()
set_target_properties(
  oneflow-lite-compile PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH "${_origin_prefix}"
                                  INSTALL_RPATH "${_origin_prefix}")

llvm_update_compile_flags(oneflow-lite-compile)

target_link_libraries(oneflow-lite-compile PRIVATE OneFlowLiteConversion ${dialect_libs}
                                                   flatcc-runtime)
