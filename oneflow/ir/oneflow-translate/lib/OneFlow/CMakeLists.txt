add_llvm_library(MLIROneFlowTranslate
        MLIROneFlowTranslate.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/oneflow-translate/include/OneFlow

        DEPENDS
        OneFlowTranslateRewritesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        )
llvm_update_compile_flags(MLIROneFlowTranslate)

target_link_libraries(MLIROneFlowTranslate
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

mlir_check_link_libraries(MLIROneFlowTranslate)
