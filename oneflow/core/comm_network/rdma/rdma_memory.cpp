#include "oneflow/core/comm_network/rdma/rdma_memory.h"
#include "glog/logging.h"

namespace oneflow {

RdmaMem::RdmaMem(ibv_pd* pd) : pd_(pd), is_registered_(false), mr_(nullptr) {}

RdmaMem::~RdmaMem() {
  if (is_registered_ == true) { Unregister(); }
}

void RdmaMem::Register(void* mem_ptr, size_t byte_size) {
  CHECK(pd_);
  mr_ = ibv_reg_mr(pd_, mem_ptr, byte_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_REMOTE_READ);
  CHECK(mr_);

  sge_.addr = reinterpret_cast<int64_t>(mem_ptr);
  sge_.length = byte_size;
  sge_.lkey = mr_->lkey;

  is_registered_ = true;
}

void RdmaMem::Unregister() {
  CHECK_EQ(ibv_dereg_mr(mr_), 0);
  is_registered_ = false;
}

RdmaMemDesc RdmaMem::GetRegisteredRdmaMemDesc() {
  CHECK_EQ(is_registered_, true);
  RdmaMemDesc rdma_mem_desc;
  rdma_mem_desc.mem_ptr = reinterpret_cast<void*>(sge_.addr);
  rdma_mem_desc.byte_size = sge_.length;
  rdma_mem_desc.token = mr_->rkey;
  return rdma_mem_desc;
}

}  // namespace oneflow
