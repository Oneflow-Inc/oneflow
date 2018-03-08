#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_ENDPOINT_MANAGER_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_ENDPOINT_MANAGER_H_

#include "oneflow/core/comm_network/ibverbs/ibverbs_connection.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

#include <netdb.h>
#include <arpa/inet.h>

namespace oneflow {

class EndpointManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EndpointManager);
  EndpointManager();
  ~EndpointManager();

  IBVerbsMemDesc* NewIBVerbsMemDesc(void* mem_ptr, size_t byte_size);
  IBVerbsConnection* NewIBVerbsConnection();

  void Read(void* read_ctx, int64_t src_machine_id,
            IBVerbsMemDesc* local_mem_desc,
            IBVerbsMemDescProto& remote_mem_desc_proto);
  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg);

 private:
  void InitRdma();
  void Start();
  void Stop();
  void PollLoop();
  void PollSendQueue();
  void PollRecvQueue();
  std::tuple<ActorMsg*, IBVerbsMemDesc*> AllocateSendMsg();
  void ReleaseSendMsg(ActorMsg* msg);

  enum { kPrePostRecvNum = 15 };  // TODO
  HashMap<ActorMsg*, IBVerbsConnection*> recv_msg2conn_ptr_;
  HashMap<ActorMsg*, IBVerbsMemDesc*> recv_msg2mem_desc_;
  HashMap<int64_t, IBVerbsConnection*> connection_pool_;

  std::mutex send_msg_pool_mutex_;
  std::queue<ActorMsg*> send_msg_pool_;
  HashMap<ActorMsg*, IBVerbsMemDesc*> send_msg2mem_desc_;

  std::thread poll_thread_;
  bool poll_state_;

  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H_
