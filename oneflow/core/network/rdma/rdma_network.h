#ifndef ONEFLOW_CORE_NETWORK_RDMA_RDMA_NETWORK_H_
#define ONEFLOW_CORE_NETWORK_RDMA_RDMA_NETWORK_H_

#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/rdma/switch.h"
#include "oneflow/core/network/rdma/connection_pool.h"
#include "oneflow/core/network/rdma/message_pool.h"
#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

class RdmaNetwork final : public Network {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaNetwork);
  RdmaNetwork();
  ~RdmaNetwork();

  void Init(int64_t my_machine_id, const NetworkTopology& net_topo) override;
  void Finalize() override;

  NetworkMessage* RegsiteMessage(const ActorMsg& actor_msg) override;
  NetworkMemory* RegisteMemory(void* dptr, size_t len) override;

  void Send(const NetworkMessage& msg) override;
  void SetCallbackForReceivedActorMsg(
      std::function<void(const ActorMsg&)> callback) override;

  void Read(const MemoryDescriptor& remote_memory_descriptor,
            NetworkMemory* local_memory) override;
  
  bool Poll(NetworkResult* result) override;
  void Barrier() override;

 private:
  void InitConnections();
  Connection* NewConnection();

  // Connection establishment routine
  // NDSPI connection establishment follows an active/passive model, where
  // passive side listens for connections requests initiated by active side

  // If there is a connection between two nodes, we assume the node
  // with smaller id/rank is the active side, while the node with larger id
  // is the passive side.

  // From the topo, we can get all the connections information, and know the
  // number of nodes(connections) that one node need to listen passively,
  // as well as the number of nodes(connections) that one node need to connect
  // actively

  // As passive side, create |listener_|, bind it and start listening
  // void StartListen();
  // All the active sides connect to the passive sides
  void EstablishConnection();

  // |result| is owned by the caller, and the received message will be held in
  // result->net_msg, having result->type == NetworkResultType::kReceiveMsg.
  bool PollRecvQueue(NetworkResult* result);

  // |result| is owned by the caller, there are two types of complection events
  // for initiator:
  // (1) successfully sends out a message to a peer, in this case:
  //     result->type == NetworkResultType::kSendOk
  // (2) successfully reads a piece of data from a peer, in this case:
  //     result->type == NetworkResultType::kReadOk
  //
  // Both send request and read request are submitted to the send request queue.
  bool PollSendQueue(NetworkResult* result);

  // Post a new Request object to the receiving queue connecting to |peer_rank|
  // void PostRecvRequest(int64_t peer_machine_id);
  // Re-post the Request object indexed by |time_stamp| to the receive queue
  // connecting to |peer_rank|, just updating its time_stamp to a new value.
  // void RePostRecvRequest(int64_t peer_machine_id, int32_t time_stamp);

  const MemoryDescriptor& GetMemoryDescriptor(int64_t register_id) const;

  // As active side, try to connect to others;
  // return true if successfully, false if failed
  // bool TryConnectTo(int64_t peer_machine_id);
  // void CompleteConnectionTo(int64_t peer_machine_id);

  // As passive side, prepare for others' connect
  // int32_t WaitForConnection();

  // estimate the pre-post number
  static const int kPrePostRecvNumber = 16;

  RdmaWrapper* rdma_wrapper_;
  int64_t my_machine_id_;
  int port_;
  NetworkTopology net_topo_;

  std::shared_ptr<RequestPool> request_pool_;
  std::shared_ptr<ConnectionPool> connection_pool_;

  int32_t time_stamp_of_last_read_request_;

  // build the dict of MemoryDescriptor
  std::unordered_map<int64_t, MemoryDescriptor> register_id_to_mem_descriptor_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_RDMA_NETWORK_H_
