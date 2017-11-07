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
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

class EndpointManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EndpointManager);
  EndpointManager();
  ~EndpointManager();

  void InitRdma();
  RdmaMem* NewRdmaMem();
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

  ConnectionInfo& GetMachineConnInfo() {
    int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
    if (connection_pool_.find(this_machine_id) == connection_pool_.end()) {
      connection_pool_.emplace(this_machine_id, NewConnection());
    }
    return connection_pool_.at(this_machine_id)->mut_this_mach_conn_info();
  }

  enum { kPrePostRecvNum = 15 };  // TODO
  HashMap<const ActorMsg*, Connection*> recv_msg2conn_ptr_;
  HashMap<const ActorMsg*, RdmaMem*> recv_msg2rdma_mem_;
  HashMap<int64_t, Connection*> connection_pool_;

  std::thread thread_;
  bool thread_state_;

  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
