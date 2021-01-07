set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf

set(LLVM_TARGET_DEFINITIONS OneFlowTranslateRewrites.td)
mlir_tablegen(OneFlowTranslateRewrites.inc -gen-rewriters -I${PROJECT_SOURCE_DIR}/include/OneFlow)
add_public_tablegen_target(OneFlowTranslateRewritesIncGen)

add_llvm_executable(oneflow-translate
  oneflow-translate.cpp
  DEPENDS
  OneFlowTranslateRewritesIncGen
  )
llvm_update_compile_flags(oneflow-translate)
target_link_libraries(oneflow-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIROneFlow
  of_protoobj
  )

mlir_check_link_libraries(oneflow-translate)
