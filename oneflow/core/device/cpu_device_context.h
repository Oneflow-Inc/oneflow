#ifndef ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() = delete;
  ~CpuDeviceCtx() = default;

  CpuDeviceCtx(int64_t work_stream_id) { set_work_stream_id(work_stream_id); }

  void AddCallBack(std::function<void()> callback) const override {
    callback();
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
