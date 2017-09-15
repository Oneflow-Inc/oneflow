#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_COMM_NETWORK_H_

#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

class RdmaCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaCommNet);
  ~RdmaCommNet() = default;

  static void Init() { CommNet::set_comm_network_ptr(new RdmaCommNet); }

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

  void Read(const void* src_token, const void* dst_token,
            std::function<void()> callback) override {
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
  RdmaCommNet() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_COMM_NETWORK_H_
