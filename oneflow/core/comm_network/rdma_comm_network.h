#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_COMM_NETWORK_H_

#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

class RdmaCommNetwork final : public CommNetwork {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaCommNetwork);
  ~RdmaCommNetwork() = default;

  static void Init() { CommNetwork::set_comm_network_ptr(new RdmaCommNetwork); }

  const void* RegisterMemory(void* dptr) override {
    // TODO
    return nullptr;
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
  void Barrier(const std::string& barrier_name) override {
    // TODO
  }

 private:
  RdmaCommNetwork() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_COMM_NETWORK_H_
