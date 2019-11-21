#ifndef ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() = default;
  ~CpuDeviceCtx() = default;

  std::unique_ptr<DeviceCtx> Copy() const { return std::unique_ptr<DeviceCtx>(new CpuDeviceCtx()); }

  void AddCallBack(std::function<void()> callback) const override { callback(); }
  void AddCallBack(std::function<void()> callback, const std::string& op_name) const override {
    AddCallBack(callback);
  }

 private:
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
