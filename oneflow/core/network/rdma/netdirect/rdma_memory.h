#ifndef ONEFLOW_CORE_NETWORK_RDMA_NETDIRECT_RDMA_MEMORY_H_
#define ONEFLOW_CORE_NETWORK_RDMA_NETDIRECT_RDMA_MEMORY_H_

#include "oneflow/core/network/rdma/netdirect/ndsupport.h"
#include "oneflow/core/network/network_memory.h"

namespace oneflow {

class RdmaMemory : public NetworkMemory {
 public:
  using NetworkMemory::memory_;
  using NetworkMemory::size_;
  using NetworkMemory::descriptor_;

  RdmaMemory() = default;
  explicit RdmaMemory(IND2MemoryRegion* memory_region);
  ~RdmaMemory();

  void Register() override;
  void Unregister() override;

  void* sge() override { return &sge_; }

 private:
  ND2_SGE sge_;
  IND2MemoryRegion* memory_region_;
  RdmaMemory(const RdmaMemory&) = delete;
  void operator=(const RdmaMemory&) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_NETDIRECT_RDMA_MEMORY_H_
