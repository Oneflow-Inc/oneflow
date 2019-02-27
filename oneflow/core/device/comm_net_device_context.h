#ifndef ONEFLOW_CORE_DEVICE_COMM_NET_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_COMM_NET_DEVICE_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

class CommNetDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetDeviceCtx);
  CommNetDeviceCtx() = default;
  ~CommNetDeviceCtx() = default;

  std::unique_ptr<DeviceCtx> Copy() const { UNIMPLEMENTED(); }

  void AddCallBack(std::function<void()> callback) const override {
    Global<CommNet>::Get()->AddReadCallBack(callback);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_COMM_NET_DEVICE_CONTEXT_H_
