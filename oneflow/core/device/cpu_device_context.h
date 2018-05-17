#ifndef ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() = delete;
  ~CpuDeviceCtx() = default;

  CpuDeviceCtx(void* buf_ptr, size_t buf_size) : DeviceCtx(buf_ptr, buf_size) {}

  void AddCallBack(std::function<void()> callback) const override { callback(); }

 private:
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
