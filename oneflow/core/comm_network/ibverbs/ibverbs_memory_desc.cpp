#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

IBVerbsMemDesc::IBVerbsMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size) {
  CHECK_GE(byte_size, 1);
  size_t block_num =
      (byte_size - 1) / Global<ResourceDesc, ForSession>::Get()->rdma_mem_block_byte() + 1;
  sge_vec_.reserve(block_num);
  mr_vec_.reserve(block_num);
  char* ch_mem_ptr = reinterpret_cast<char*>(mem_ptr);
  while (byte_size > 0) {
    size_t cur_size =
        std::min<size_t>(byte_size, Global<ResourceDesc, ForSession>::Get()->rdma_mem_block_byte());
    ibv_mr* cur_mr =
        ibv_reg_mr(pd, ch_mem_ptr, cur_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    CHECK(cur_mr);
    mr_vec_.push_back(cur_mr);
    ibv_sge cur_sge;
    cur_sge.addr = reinterpret_cast<uint64_t>(ch_mem_ptr);
    cur_sge.length = cur_size;
    cur_sge.lkey = cur_mr->lkey;
    sge_vec_.push_back(cur_sge);
    ch_mem_ptr += cur_size;
    byte_size -= cur_size;
  }
  CHECK_EQ(byte_size, 0);
  CHECK_EQ(block_num, sge_vec_.size());
  CHECK_EQ(block_num, mr_vec_.size());
}

IBVerbsMemDesc::~IBVerbsMemDesc() {
  for (ibv_mr* mr : mr_vec_) { CHECK_EQ(ibv_dereg_mr(mr), 0); }
}

IBVerbsMemDescProto IBVerbsMemDesc::ToProto() {
  IBVerbsMemDescProto proto;
  for (const ibv_sge& sge : sge_vec_) { proto.add_mem_ptr(sge.addr); }
  for (ibv_mr* mr : mr_vec_) { proto.add_mr_rkey(mr->rkey); }
  return proto;
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
