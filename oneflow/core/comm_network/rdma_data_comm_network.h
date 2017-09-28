#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_DATA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_DATA_COMM_NETWORK_H_

#include "oneflow/core/comm_network/data_comm_network.h"

namespace oneflow {

class RdmaDataCommNet final : public DataCommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaDataCommNet);
  ~RdmaDataCommNet() = default;

  static void Init() { DataCommNet::set_comm_network_ptr(new RdmaDataCommNet); }

  const void* RegisterMemory(void* dptr, size_t byte_size) override {
    // TODO
    return nullptr;
  }
  void UnRegisterMemory(const void* comm_net_token) override {
    // TODO
  }
  void RegisterMemoryDone() override {
    // TODO
  }

  void* Read(int64_t src_machine_id, const void* src_token,
             const void* dst_token) override {
    // TODO
    return nullptr;
  }

  void AddReadCallBack(void* read_id, std::function<void()> callback) override {
    // TODO
  }

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override {
    // TODO
  }

 private:
  RdmaDataCommNet() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_DATA_COMM_NETWORK_H_
