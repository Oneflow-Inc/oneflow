#ifndef ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

class KernelCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(KernelCtx);
  virtual ~KernelCtx() = default;

  Channel<std::function<void()>>* cpu_stream() const { return cpu_stream_; }
  const cudaStream_t& cuda_stream() const { return *cuda_stream_; }

  virtual void AddCallBack(std::function<void()>) const = 0;

 protected:
  KernelCtx() : cpu_stream_(nullptr), cuda_stream_(nullptr) {}

  void set_cpu_stream(Channel<std::function<void()>>* val) {
    cpu_stream_ = val;
  }
  void set_cuda_stream(const cudaStream_t* val) {
    cuda_stream_ = val;
  }

 private:
  Channel<std::function<void()>>* cpu_stream_;
  const cudaStream_t* cuda_stream_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
