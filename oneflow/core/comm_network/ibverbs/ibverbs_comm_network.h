#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_qp.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

class IBVerbsCommNet final : public CommNetIf<IBVerbsMemDesc> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsCommNet);
  IBVerbsCommNet() = delete;
  ~IBVerbsCommNet();

  static void Init(const Plan& plan) { Global<CommNet>::SetAllocated(new IBVerbsCommNet(plan)); }

  void RegisterMemoryDone() override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  IBVerbsMemDesc* NewMemDesc(void* ptr, size_t byte_size) override {
    return new IBVerbsMemDesc(pd_, ptr, byte_size);
  }

  IBVerbsCommNet(const Plan&);
  void DoRead(int64_t stream_id, int64_t src_machine_id, void* src_token, void* dst_token) override;
  void PollCQ();
  void InitContext(const std::string&);
  uint32_t QueryPort(uint32_t, ibv_port_attr*);
  uint32_t QueryGid(uint32_t, uint32_t, ibv_port_attr*, ibv_gid*);

  static const int32_t max_poll_wc_num_;

  std::vector<HashMap<void*, IBVerbsMemDescProto>> token2mem_desc_;
  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* cq_;
  std::vector<IBVerbsQP*> qp_vec_;
  std::atomic_flag poll_exit_flag_;
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
