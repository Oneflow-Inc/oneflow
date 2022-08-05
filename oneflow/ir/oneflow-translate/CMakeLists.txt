set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf

include_directories(${PROJECT_SOURCE_DIR}/oneflow-translate/include)
include_directories(${PROJECT_BINARY_DIR}/oneflow-translate/include)

add_subdirectory(include)
add_subdirectory(lib)

add_llvm_executable(oneflow-translate oneflow-translate.cpp DEPENDS MLIROneFlow
                    MLIROneFlowTranslation)

set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()
set_target_properties(
  oneflow-translate PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH "${_origin_prefix}"
                               INSTALL_RPATH "${_origin_prefix}")

llvm_update_compile_flags(oneflow-translate)

target_link_libraries(oneflow-translate PRIVATE ${dialect_libs} ${translation_libs}
                      PUBLIC MLIRTranslateLib MLIROneFlowTranslation)

mlir_check_link_libraries(oneflow-translate)
