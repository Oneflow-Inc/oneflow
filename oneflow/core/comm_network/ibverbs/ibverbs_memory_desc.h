#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc_proto.pb.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include <infiniband/verbs.h>

namespace oneflow {

class IBVerbsMemDesc {
 public:
  IBVerbsMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size);
  ~IBVerbsMemDesc() { CHECK_EQ(ibv_dereg_mr(mr_), 0); }

  IBVerbsMemDescProto IBVerbsMemDescToProto();
  ibv_sge* ibv_sge_ptr() { return &sge_; }

 private:
  ibv_sge sge_;
  ibv_mr* mr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_
