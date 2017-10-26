#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H

#include <infiniband/verbs.h>
#include <netdb.h>
#include <string>
#include "oneflow/core/comm_network/rdma/conn_info.pb.h"
#include "oneflow/core/comm_network/rdma/connection.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class EndpointManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EndpointManager);
  EndpointManager() = default;
  ~EndpointManager();

  void Init(const std::string& my_ip, int32_t my_port);
  void InitRdma();
  RdmaMem* NewRdmaMem();
  Connection* NewConnection();
  ibv_qp* NewQueuePair();

  void Read(void* read_ctx, int64_t src_machine_id, const RdmaMem* local_mem,
            const RdmaMemDesc& remote_mem_desc);
  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg);

  void Start();
  void Stop();

 private:
  void PollLoop();
  void PollSendQueue();
  void PollRecvQueue();

  ConnectionInfo& GetMachineConnInfo() { return conn_info_; }

  enum { kPrePostRecvNum = 15 };  // TODO
  ConnectionInfo conn_info_;
  HashMap<ActorMsg*, RdmaMem*> recv_msg2rdma_mem_;
  HashMap<int64_t, Connection*> connection_pool_;

  std::thread thread_;
  bool thread_state_;

  ibv_context* context_;
  enum ibv_mtu active_mtu_;
  ibv_pd* pd_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
