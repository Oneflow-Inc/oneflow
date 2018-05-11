#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs.pb.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include <infiniband/verbs.h>

namespace oneflow {

class IBVerbsMemDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsMemDesc);
  IBVerbsMemDesc() = delete;
  IBVerbsMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size);
  ~IBVerbsMemDesc() { PCHECK(ibv_dereg_mr(mr_) == 0); }

  const ibv_sge& ibv_sge() { return sge_; }

  void ToProto(IBVerbsMemDescProto* proto);

 private:
  ibv_sge sge_;
  ibv_mr* mr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_
