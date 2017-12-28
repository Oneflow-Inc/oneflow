#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_H

#include "oneflow/core/common/util.h"

#ifdef WITH_RDMA

#include <infiniband/verbs.h>
#include "oneflow/core/comm_network/rdma/rdma_memory_desc.pb.h"

namespace oneflow {

class RdmaMem {
 public:
  RdmaMem(ibv_pd* pd, void* mem_ptr, size_t byte_size);
  ~RdmaMem() { CHECK_EQ(ibv_dereg_mr(mr_), 0); }

  RdmaMemDesc GenRdmaMemDesc();
  ibv_sge* ibv_sge_ptr() { return &sge_; }

 private:
  ibv_sge sge_;
  ibv_mr* mr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_H
