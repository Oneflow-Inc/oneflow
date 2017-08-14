#ifndef ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_MEMORY_H_
#define ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_MEMORY_H_

#include <infiniband/verbs.h>
#include "oneflow/core/network/network_memory.h"

namespace oneflow {

class RdmaMemory final : public NetworkMemory {
 public:
  using NetworkMemory::descriptor_;
  using NetworkMemory::memory_;
  using NetworkMemory::size_;

  explicit RdmaMemory(ibv_pd* protect_domain);
  ~RdmaMemory();

  void Register() override;
  void Unregister() override;

  void* sge() override { return &sge_; }

 private:
  ibv_sge sge_;
  ibv_pd* protect_domain_;  // Not owned
  ibv_mr* memory_region_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_MEMORY_H_
