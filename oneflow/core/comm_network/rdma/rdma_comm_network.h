#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_COMM_NETWORK_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_COMM_NETWORK_H

#include <mutex>
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/rdma/connection_pool.h"
#include "oneflow/core/comm_network/rdma/endpoint_manager.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

class RdmaCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaCommNet);
  ~RdmaCommNet();

  static RdmaCommNet* Singleton() {
    return static_cast<RdmaCommNet*>(CommNet::Singleton());
  }

  static void Init();

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* NewActorReadId();
  void DeleteActorReadId(void* actor_read_id);
  void* Read(void* actor_read_id, int64_t src_machine_id, const void* src_token,
             const void* dst_token) override;
  void AddReadCallBack(void* actor_read_id, void* read_id,
                       std::function<void()> callback) override;
  void AddReadCallBackDone(void* actor_read_id, void* read_id) override;
  void ReadDone(void* read_done_id);

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  struct ReadContext {
    std::list<std::function<void()>> cbl;
    std::mutex done_cnt_mtx;
    int8_t done_cnt;
  };
  struct ActorReadContext {
    std::mutex read_ctx_list_mtx;
    std::list<ReadContext*> read_ctx_list;
  };
  Connection* NewConnection();
  RdmaCommNet();
  int8_t IncreaseDoneCnt(ReadContext*);
  void FinishOneReadContext(ActorReadContext*, ReadContext*);
  void InitRdma();

  std::mutex mem_mutex_;
  std::list<RdmaMem*> mems_;
  size_t unregister_mems_cnt_;

  std::unique_ptr<EndpointManager> endpoint_manager_;
  std::unique_ptr<ConnectionPool> connection_pool_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_RDMA_COMM_NETWORK_H
