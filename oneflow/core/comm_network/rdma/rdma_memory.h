#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_H

#include <infiniband/verbs.h>

namespace oneflow {

struct RdmaMemDesc {
  uint64_t mem_addr;
  size_t byte_size;
  uint32_t token;
};

class RdmaMem {
 public:
  RdmaMem(ibv_pd* pd);
  ~RdmaMem();

  void Register(void* mem_ptr, size_t byte_size);
  void Unregister();
  RdmaMemDesc GetRegisteredRdmaMemDesc();

  ibv_sge* ibv_sge_ptr() { return &sge_; }

 private:
  bool is_registered_;
  ibv_sge sge_;
  ibv_pd* pd_;
  ibv_mr* mr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_H
