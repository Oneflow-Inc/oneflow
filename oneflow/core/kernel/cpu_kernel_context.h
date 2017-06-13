#ifndef ONEFLOW_CORE_KERNEL_CPU_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_KERNEL_CPU_KERNEL_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CpuKernelCtx final : public KernelCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CpuKernelCtx);
  CpuKernelCtx() = delete;
  ~CpuKernelCtx() = default;
  
  CpuKernelCtx(Channel<std::function<void()>>* chan) {
    set_cpu_stream(chan);
  }

  void AddCallBack(std::function<void()> callback) const override {
    cpu_stream()->Send(callback);
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_CPU_KERNEL_CONTEXT_H_
