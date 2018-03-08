#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/memory_desc_manager.h"
#include "oneflow/core/comm_network/ibverbs/endpoint_manager.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

class IBVerbsCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsCommNet);
  ~IBVerbsCommNet() = default;

  static IBVerbsCommNet* Singleton() {
    return static_cast<IBVerbsCommNet*>(CommNet::Singleton());
  }

  static void Init();

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  IBVerbsCommNet() = default;
  void DoRead(void* read_id, int64_t src_machine_id, const void* src_token,
              const void* dst_token) override;

  MemDescMgr<IBVerbsMemDesc> mem_desc_mgr_;
  EndpointManager endpoint_manager_;
  HashMap<uint64_t, IBVerbsMemDescProto> token2mem_desc_proto_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
