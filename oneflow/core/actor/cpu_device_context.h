#ifndef ONEFLOW_CORE_ACTOR_CPU_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_ACTOR_CPU_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() = delete;
  ~CpuDeviceCtx() = default;

  CpuDeviceCtx(Channel<std::function<void()>>* chan) { set_cpu_stream(chan); }

  void AddCallBack(std::function<void()> callback) const override {
    cpu_stream()->Send(callback);
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_CPU_DEVICE_CONTEXT_H_
