#ifndef ONEFLOW_CORE_DEVICE_COMM_NET_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_COMM_NET_DEVICE_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

class CommNetDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetDeviceCtx);
  CommNetDeviceCtx(int64_t local_stream_id) : local_stream_id_(local_stream_id){};
  ~CommNetDeviceCtx() = default;

  std::unique_ptr<DeviceCtx> Copy() const { UNIMPLEMENTED(); }

  void AddCallBack(std::function<void()> callback) const override {
    Global<CommNet>::Get()->AddReadCallBack(local_stream_id_, callback);
  }

 private:
  int64_t local_stream_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_COMM_NET_DEVICE_CONTEXT_H_
