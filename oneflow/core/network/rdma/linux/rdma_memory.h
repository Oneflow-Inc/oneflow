#ifndef ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_MEMORY_H_
#define ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_MEMORY_H_

#include <infiniband/verbs.h>
#include "oneflow/core/network/network_memory.h"

namespace oneflow {

class RdmaMemory : public NetworkMemory {
 public:
  using NetworkMemory::memory_;
  using NetworkMemory::size_;
  using NetworkMemory::descriptor_;

  RdmaMemory() = default;
  RdmaMemory(struct ibv_mr* memory_region, struct ibv_pd* protect_domain);

  void Register() override;
  void Unregister() override;

  void* sge() override { return &sge_; }

 private:
  struct ibv_sge sge_;
  struct ibv_pd* protect_domain_;
  struct ibv_mr* memory_region_;
  RdmaMemory(const RdmaMemory&) = delete;
  void operator=(const RdmaMemory&) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_MEMORY_H_
