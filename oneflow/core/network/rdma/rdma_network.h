#ifndef ONEFLOW_CORE_NETWORK_RDMA_RDMA_NETWORK_H_
#define ONEFLOW_CORE_NETWORK_RDMA_RDMA_NETWORK_H_

#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/rdma/connection_pool.h"
#include "oneflow/core/network/rdma/message_pool.h"
#include "oneflow/core/network/rdma/request_pool.h"
#include "oneflow/core/network/rdma/switch.h"

namespace oneflow {

class RdmaNetwork final : public Network {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaNetwork);
  RdmaNetwork();
  ~RdmaNetwork() = default;

  void Init(int64_t my_machine_id, const NetworkTopology& net_topo) override;

  NetworkMemory* RegisterMemory(void* dptr, size_t len) override;

  void SendMsg(const NetworkMessage& msg) override;
  void SetCallbackForReceivedActorMsg(std::function<void()> callback) override;
  void Read(const MemoryDescriptor& remote_memory_descriptor,  // TODO(shiyuan)
            NetworkMemory* local_memory,
            std::function<void()> callback) override;

  bool Poll(NetworkResult* result) override;
  void Barrier() override;

 private:
  Connection* NewConnection(const std::string& my_ip, int32_t my_port);
  // passive side listens for connections requests initiated by active
  // side(smaller id/rank)
  void EstablishConnection();

  // |result| is owned by the caller, and the received message will be held in
  // result->net_msg, having result->type == NetworkResultType::kReceiveMsg.
  bool PollRecvQueue(NetworkResult* result);
  // |result| is owned by the caller.
  // Both send request and read request are submitted to the send request queue.
  bool PollSendQueue(NetworkResult* result);

  // estimate the pre-post number
  static const int kPrePostRecvNumber = 15;  // TODO(shiyuan)
  NetworkTopology net_topo_;
  int64_t my_machine_id_;
  int port_;

  std::unique_ptr<EndpointManager> endpoint_manager_;
  std::unique_ptr<RequestPool> request_pool_;
  std::unique_ptr<ConnectionPool> connection_pool_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_RDMA_NETWORK_H_
