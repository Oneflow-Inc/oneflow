#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_DATA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_DATA_COMM_NETWORK_H_

#include "oneflow/core/comm_network/data_comm_network.h"

namespace oneflow {

class RdmaDataCommNet final : public DataCommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaDataCommNet);
  ~RdmaDataCommNet() = default;

  static void Init() { DataCommNet::set_comm_network_ptr(new RdmaDataCommNet); }

  const void* RegisterMemory(void* dptr) override {
    // TODO
    return nullptr;
  }
  void UnRegisterMemory(const void* comm_net_token) override {
    // TODO
  }
  void RegisterMemoryDone() override {
    // TODO
  }

  void* CreateStream() override {
    // TODO
    return nullptr;
  }
  void Read(void* stream_id, const void* src_token, const void* dst_token) {
    // TODO
  }
  void AddCallBack(void* stream_id, std::function<void()>) {
    // TODO
  }

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override {
    // TODO
  }

  void SetCallbackForReceivedActorMsg(
      std::function<void(const ActorMsg&)> callback) override {
    // TODO
  }

 private:
  RdmaDataCommNet() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_DATA_COMM_NETWORK_H_
