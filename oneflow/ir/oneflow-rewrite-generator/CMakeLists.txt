set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(oneflow-rewrite-generator generator.cpp)

set_target_properties(
  oneflow-rewrite-generator PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH "${_origin_prefix}"
                               INSTALL_RPATH "${_origin_prefix}")

llvm_update_compile_flags(oneflow-rewrite-generator)

target_compile_features(oneflow-rewrite-generator PRIVATE cxx_std_17)

target_link_libraries(oneflow-rewrite-generator PRIVATE ${dialect_libs} ${translation_libs}
                      PUBLIC oneflow_cpp MLIRTranslateLib MLIROneFlowExtension)

mlir_check_link_libraries(oneflow-rewrite-generator)
