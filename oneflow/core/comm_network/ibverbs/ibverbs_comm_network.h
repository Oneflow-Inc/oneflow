#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/memory_desc_manager.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_qp.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include <netdb.h>
#include <arpa/inet.h>

namespace oneflow {

class IBVerbsCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsCommNet);
  IBVerbsCommNet() = delete;
  ~IBVerbsCommNet();

  static void Init(const Plan& plan) { Global<CommNet>::SetAllocated(new IBVerbsCommNet(plan)); }

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  IBVerbsCommNet(const Plan&);
  void DoRead(ReadContext* read_ctx, int64_t src_machine_id, const void* src_token,
              const void* dst_token) override;
  void PollCQ();

  MemDescMgr<IBVerbsMemDesc> mem_desc_mgr_;
  HashMap<const void*, IBVerbsMemDescProto> token2mem_desc_;
  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* cq_;
  std::vector<IBVerbsQP*> qp_vec_;
  std::thread poll_thread_;
};

template<>
class Global<IBVerbsCommNet> final {
 public:
  static IBVerbsCommNet* Get() { return static_cast<IBVerbsCommNet*>(Global<CommNet>::Get()); }
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
