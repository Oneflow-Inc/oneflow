if(WITH_MLIR_CUDA_CODEGEN)
  set(MLIR_RUNTIME_GPU_LIBS -Wl,--no-as-needed mlir_cuda_runtime
                            -Wl,--as-needed)
endif(WITH_MLIR_CUDA_CODEGEN)
set(MLIR_RUNTIME_LIBS -Wl,--no-as-needed mlir_c_runner_utils -Wl,--as-needed)
oneflow_add_mlir_library(MLIROneFlowRuntime Runtime.cpp)
target_link_libraries(MLIROneFlowRuntime PUBLIC ${MLIR_RUNTIME_LIBS}
                                                ${MLIR_RUNTIME_GPU_LIBS})
