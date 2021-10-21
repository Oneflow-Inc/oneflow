set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf
function(add_llvm_static_library)
  set(BUILD_SHARED_LIBS_TMP ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  add_llvm_library(${ARGV})
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_TMP})
endfunction()
add_llvm_static_library(MLIROneFlowJIT
  JIT.cpp
  DEPENDS
	LINK_LIBS PUBLIC
	MLIRIR
  BUILDTREE_ONLY
)

llvm_update_compile_flags(MLIROneFlowJIT)

if(BUILD_CUDA)
  set(MLIR_RUNTIME_GPU_LIBS -Wl,--no-as-needed mlir_cuda_runtime -Wl,--as-needed)
endif(BUILD_CUDA)

target_link_libraries(MLIROneFlowJIT
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
)

mlir_check_link_libraries(MLIROneFlowJIT)
