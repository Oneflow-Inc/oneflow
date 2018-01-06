#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_COMM_NETWORK_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_COMM_NETWORK_H

#ifdef WITH_RDMA

#include <mutex>
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/rdma/endpoint_manager.h"

namespace oneflow {

class RdmaCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaCommNet);
  ~RdmaCommNet();

  static RdmaCommNet* Singleton() {
    return static_cast<RdmaCommNet*>(CommNet::Singleton());
  }

  static void Init();

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* Read(void* actor_read_id, int64_t src_machine_id, const void* src_token,
             const void* dst_token) override;
  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  RdmaCommNet();

  std::mutex mem_mutex_;
  std::list<RdmaMem*> mems_;
  size_t unregister_mems_cnt_;

  EndpointManager* endpoint_manager_;
  HashMap<uint64_t, RdmaMemDesc> token2mem_desc_;
};

}  // namespace oneflow

#endif  // WITH_RDMA

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_COMM_NETWORK_H
