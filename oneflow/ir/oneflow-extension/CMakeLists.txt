set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf
oneflow_add_llvm_library(MLIROneFlowExtension
  extension.cpp
  ir_pass.cpp
  DEPENDS
	LINK_LIBS PUBLIC
	MLIRIR
  BUILDTREE_ONLY
)

llvm_update_compile_flags(MLIROneFlowExtension)

if(WITH_MLIR_CUDA_CODEGEN)
  set(MLIR_RUNTIME_GPU_LIBS -Wl,--no-as-needed mlir_cuda_runtime -Wl,--as-needed)
endif(WITH_MLIR_CUDA_CODEGEN)
set(MLIR_RUNTIME_LIBS -Wl,--no-as-needed mlir_c_runner_utils -Wl,--as-needed)
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
  MLIRExecutionEngine
  MLIROneFlowTranslation
  oneflow
  PUBLIC
  ${MLIR_RUNTIME_LIBS}
  ${MLIR_RUNTIME_GPU_LIBS}
)

if (BUILD_SHARED_LIBS)
  get_filename_component(ONEFLOW_BUILD_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../.. ABSOLUTE)
  get_property(TRANSLATE_INSTALL_RPATH TARGET MLIROneFlowExtension PROPERTY INSTALL_RPATH)
  list(APPEND TRANSLATE_INSTALL_RPATH ${PROTOBUF_LIBRARY_DIR})
  list(APPEND TRANSLATE_INSTALL_RPATH ${ONEFLOW_BUILD_ROOT_DIR})
  set_target_properties(MLIROneFlowExtension PROPERTIES INSTALL_RPATH "${TRANSLATE_INSTALL_RPATH}")
endif()

mlir_check_link_libraries(MLIROneFlowExtension)
add_custom_target(mex DEPENDS MLIROneFlowExtension)
