oneflow_add_mlir_library(MLIROneFlowRuntime Runtime.cpp)
if(WITH_MLIR_CUDA_CODEGEN)
  set(MLIR_RUNTIME_GPU_LIBS mlir_cuda_runtime)
endif(WITH_MLIR_CUDA_CODEGEN)
target_link_libraries(MLIROneFlowRuntime PUBLIC -Wl,--no-as-needed ${MLIR_RUNTIME_GPU_LIBS}
                                                mlir_c_runner_utils -Wl,--as-needed)
