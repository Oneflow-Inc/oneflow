set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf
add_llvm_library(MLIROneFlowExtension
  extension.cpp
  DEPENDS
	LINK_LIBS PUBLIC
	MLIRIR
  BUILDTREE_ONLY
)

llvm_update_compile_flags(MLIROneFlowExtension)

target_link_libraries(MLIROneFlowExtension
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
  of_ccobj
)

if (BUILD_SHARED_LIBS)
  get_filename_component(ONEFLOW_BUILD_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../.. ABSOLUTE)
  get_property(TRANSLATE_INSTALL_RPATH TARGET MLIROneFlowExtension PROPERTY INSTALL_RPATH)
  list(APPEND TRANSLATE_INSTALL_RPATH ${PROTOBUF_LIBRARY_DIR})
  list(APPEND TRANSLATE_INSTALL_RPATH ${ONEFLOW_BUILD_ROOT_DIR})
  set_target_properties(MLIROneFlowExtension PROPERTIES INSTALL_RPATH "${TRANSLATE_INSTALL_RPATH}")
endif()

mlir_check_link_libraries(MLIROneFlowExtension)
