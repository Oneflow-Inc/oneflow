add_llvm_library(MLIROneFlowTranslation
        MLIROneFlowTranslation.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/oneflow-translate/include/OneFlow

        DEPENDS

	LINK_LIBS PUBLIC
	MLIRIR
  BUILDTREE_ONLY
        )
llvm_update_compile_flags(MLIROneFlowTranslation)

target_link_libraries(MLIROneFlowTranslation
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

if (BUILD_SHARED_LIBS)
  get_filename_component(ONEFLOW_BUILD_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../.. ABSOLUTE)
  get_property(TRANSLATE_INSTALL_RPATH TARGET MLIROneFlowTranslation PROPERTY INSTALL_RPATH)
  list(APPEND TRANSLATE_INSTALL_RPATH ${PROTOBUF_LIBRARY_DIR})
  list(APPEND TRANSLATE_INSTALL_RPATH ${ONEFLOW_BUILD_ROOT_DIR})
  set_target_properties(MLIROneFlowTranslation PROPERTIES INSTALL_RPATH "${TRANSLATE_INSTALL_RPATH}")
endif()

mlir_check_link_libraries(MLIROneFlowTranslation)
