#ifndef ONEFLOW_NETWORK_RDMA_LINUX_MEMORY_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_MEMORY_H_

#include <rdma/rdma_cma.h>
#include "network/network_memory.h"

namespace oneflow{

class Memory : public NetworkMemory {
public:
  using NetworkMemory::memory_;
  using NetworkMemory::size_;
  using NetworkMemory::descriptor_;

  explicit Memory(ibv_mr* memory_region) {
    memory_region_ = memory_region;
  }

  //Register as RDMA memory region
  void Register() override {
    
  }

  void Unregister() override {

  }

  const void* sge() const override { return &sge_; }

private:
  ibv_sge sge_;
  ibv_mr* memory_region_;
  Memory(const Memory&) = delete;
  void operator=(const Memory&) = delete;
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_MEMORY_H_
