#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H

#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

#ifdef WITH_RDMA

#include <netdb.h>
#include <arpa/inet.h>
#include "oneflow/core/comm_network/rdma/connection.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"

namespace oneflow {

class EndpointManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EndpointManager);
  EndpointManager();
  ~EndpointManager();

  void InitRdma();
  RdmaMem* NewRdmaMem(void* mem_ptr, size_t byte_size);
  Connection* NewConnection();

  void Read(void* read_ctx, int64_t src_machine_id, const RdmaMem* local_mem,
            const RdmaMemDesc& remote_mem_desc);
  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg);

  void Start();
  void Stop();

 private:
  void PollLoop();
  void PollSendQueue();
  void PollRecvQueue();
  std::tuple<ActorMsg*, RdmaMem*> AllocateSendMsg();
  void ReleaseSendMsg(ActorMsg* msg);

  enum { kPrePostRecvNum = 15 };  // TODO
  HashMap<const ActorMsg*, Connection*> recv_msg2conn_ptr_;
  HashMap<const ActorMsg*, const RdmaMem*> recv_msg2rdma_mem_;
  HashMap<int64_t, Connection*> connection_pool_;

  std::mutex send_msg_pool_mutex_;
  std::queue<ActorMsg*> send_msg_pool_;
  HashMap<ActorMsg*, RdmaMem*> send_msg2rdma_mem_;

  std::thread thread_;
  bool thread_state_;

  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // WITH_RDMA

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
