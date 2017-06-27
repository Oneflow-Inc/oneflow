#include "oneflow/core/network/rdma/linux/rdma_memory.h"
#include "oneflow/core/network/rdma/linux/interface.h"

namespace oneflow {

RdmaMemory::RdmaMemory(struct ibv_mr* memory_region,
                       struct ibv_pd* protect_domain) {
  memory_region_ = memory_region;
  protect_domain_ = protect_domain;
}

void RdmaMemory::Register() {
  memory_region_ = ibv_reg_mr(
      protect_domain_,
      memory_,
      size_,
      IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);

  sge_.addr = (uint64_t)memory_;
  sge_.length = size_;
  sge_.lkey = memory_region_->lkey;

  descriptor_.address = (uint64_t)memory_;
  descriptor_.remote_token = memory_region_->rkey;

  if (memory_region_ != NULL) {
    registered_ = true;
  }
}

void RdmaMemory::Unregister() {
  bool res = ibv_dereg_mr(memory_region_);
  if (res == 0) {
    registered_ = false;
  }
}

}  // namespace oneflow
