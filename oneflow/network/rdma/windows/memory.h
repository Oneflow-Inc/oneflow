#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_MEMORY_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_MEMORY_H_

#include "network/rdma/windows/ndsupport.h"
#include "network/network_memory.h"

namespace oneflow {

class Memory : public NetworkMemory {
 public:
  using NetworkMemory::memory_;
  using NetworkMemory::size_;
  using NetworkMemory::descriptor_;

  Memory() = default;
  explicit Memory(IND2MemoryRegion* memory_region);

  void Register() override;
  void Unregister() override;

  const void* sge() const override { return &sge_; }

 private:
  ND2_SGE sge_;
  IND2MemoryRegion* memory_region_;
  Memory(const Memory&) = delete;
  void operator=(const Memory&) = delete;
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_MEMORY_H_
