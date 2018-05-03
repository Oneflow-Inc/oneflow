#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

void CUDART_CB CudaCallBackHandle(cudaStream_t, cudaError_t status, void* void_ptr) {
  CudaCheck(status);
  auto callback_ptr = static_cast<std::function<void()>*>(void_ptr);
  (*callback_ptr)();
  delete callback_ptr;
}

}  // namespace

void CudaDeviceCtx::AddCallBack(std::function<void()> callback_stack) const {
  auto callback_heap = new std::function<void()>(callback_stack);
  CudaCheck(cudaStreamAddCallback(cuda_stream(), &CudaCallBackHandle, callback_heap, 0));
}

#endif  // WITH_CUDA

}  // namespace oneflow
