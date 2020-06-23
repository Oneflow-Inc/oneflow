#ifndef ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/common/callback.msg.h"
#include "oneflow/core/vm/cuda_allocator.h"

namespace oneflow {
namespace vm {

#ifdef WITH_CUDA

class CudaStreamHandleDeviceCtx : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamHandleDeviceCtx);
  CudaStreamHandleDeviceCtx() = delete;
  ~CudaStreamHandleDeviceCtx() override = default;

  CudaStreamHandleDeviceCtx(CallbackMsgListPtr callback_msg_list, int64_t device_id)
      : cuda_handler_(new CudaStreamHandle(nullptr)),
        callback_msg_list_(callback_msg_list),
        cuda_allocator_(device_id) {}

  const cudaStream_t& cuda_stream() const override { return *(cuda_handler_->cuda_stream()); }
  const cublasHandle_t& cublas_pmh_handle() const override {
    return *(cuda_handler_->cublas_pmh_handle());
  }
  const cublasHandle_t& cublas_tensor_op_math_handle() const override {
    return *(cuda_handler_->cublas_tensor_op_math_handle());
  }
  const cublasHandle_t& cublas_pmd_handle() const override {
    return *(cuda_handler_->cublas_pmd_handle());
  }
  const cudnnHandle_t& cudnn_handle() const override { return *(cuda_handler_->cudnn_handle()); }

  void SyncDevice() override { CudaCheck(cudaStreamSynchronize(cuda_stream())); }

  void AddCallBack(std::function<void()> callback) const override {
    callback_msg_list_->EmplaceBack(ObjectMsgPtr<CallbackMsg>::New(callback));
  }

  vm::Allocator* mut_allocator() override { return &cuda_allocator_; }

 protected:
  std::unique_ptr<CudaStreamHandle> cuda_handler_;
  CallbackMsgListPtr callback_msg_list_;
  CudaAllocator cuda_allocator_;
};

#endif  // WITH_CUDA
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_DEVICE_CONTEXT_H_
