#include "oneflow/core/comm_network/rdma/rdma_memory.h"

#ifdef WITH_RDMA

namespace oneflow {

RdmaMem::RdmaMem(ibv_pd* pd, void* mem_ptr, size_t byte_size) {
  mr_ = ibv_reg_mr(pd, mem_ptr, byte_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_REMOTE_READ);
  CHECK(mr_);

  sge_.addr = reinterpret_cast<uint64_t>(mem_ptr);
  sge_.length = byte_size;
  sge_.lkey = mr_->lkey;
}

RdmaMemDesc RdmaMem::GetRdmaMemDesc() {
  RdmaMemDesc rdma_mem_desc;
  rdma_mem_desc.set_mem_ptr(reinterpret_cast<uint64_t>(sge_.addr));
  rdma_mem_desc.set_mr_rkey(mr_->rkey);
  return rdma_mem_desc;
}

}  // namespace oneflow

#endif  // WITH_RDMA
