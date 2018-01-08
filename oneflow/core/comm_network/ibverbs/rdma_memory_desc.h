#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_DESC_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_DESC_H_

#include "oneflow/core/common/util.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include <infiniband/verbs.h>
#include "oneflow/core/comm_network/ibverbs/rdma_memory_desc_proto.pb.h"

namespace oneflow {

class RdmaMemDesc {
 public:
  RdmaMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size);
  ~RdmaMemDesc() { CHECK_EQ(ibv_dereg_mr(mr_), 0); }

  RdmaMemDescProto GenRdmaMemDescProto();
  ibv_sge* ibv_sge_ptr() { return &sge_; }

 private:
  ibv_sge sge_;
  ibv_mr* mr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_MEMORY_DESC_H_
