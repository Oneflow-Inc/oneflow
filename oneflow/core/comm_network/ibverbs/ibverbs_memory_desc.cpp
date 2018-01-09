#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

IBVerbsMemDesc::IBVerbsMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size) {
  mr_ = ibv_reg_mr(pd, mem_ptr, byte_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_REMOTE_READ);
  CHECK(mr_);

  sge_.addr = reinterpret_cast<uint64_t>(mem_ptr);
  sge_.length = byte_size;
  sge_.lkey = mr_->lkey;
}

IBVerbsMemDescProto IBVerbsMemDesc::GenIBVerbsMemDescProto() {
  IBVerbsMemDescProto ibverbs_mem_desc_proto;
  ibverbs_mem_desc_proto.set_mem_ptr(reinterpret_cast<uint64_t>(sge_.addr));
  ibverbs_mem_desc_proto.set_mr_rkey(mr_->rkey);
  return ibverbs_mem_desc_proto;
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
