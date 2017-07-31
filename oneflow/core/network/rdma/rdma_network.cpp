#include "oneflow/core/network/rdma/rdma_network.h"

namespace oneflow {

RdmaNetwork::RdmaNetwork() 
  : rdma_wrapper_(nullptr),
    my_machine_id_(-1),
    port_(-1) {}

RdmaNetwork::~RdmaNetwork() {
  Finalize();
}

void RdmaNetwork::Init(int64_t my_machine_id, const NetworkTopology& net_topo) {
  my_machine_id_ = my_machine_id;
  port_ = net_topo.all_nodes[my_machine_id].port;
  net_topo_ = net_topo;
  rdma_wrapper_.reset(new RdmaWrapper());
  request_pool_.reset(new RequestPool());
  connection_pool_.reset(new ConnectionPool());
  rdma_wrapper_->Init(net_topo.all_nodes[my_machine_id].address.c_str(), port_);
  EstablishConnection();
}

void RdmaNetwork::Finalize() {
  register_id_to_mem_descriptor_.clear();
}

NetworkMemory* RdmaNetwork::RegisterMemory(void* dptr, size_t len,
                                           int64_t register_id) {
  NetworkMemory* net_memory = rdma_wrapper_->NewNetworkMemory();  // TODO(shiyuan)
  net_memory->Reset(dptr, len, register_id);
  return net_memory;
}

// |msg| contains src_machine_id and dst_machine_id
void RdmaNetwork::SendMessage(const NetworkMessage& msg) {
  int64_t dst_machine_id = msg.dst_machine_id;
  Connection* conn = connection_pool_->GetConnection(dst_machine_id);
  CHECK(conn);

  Request* send_request = request_pool_->AllocRequest(true);
  CHECK(send_request);

  send_request->rdma_msg->mutable_msg() = msg;

  conn->PostSendRequest(*send_request);
}

void RdmaNetwork::SetCallbackForReceivedActorMsg(
    std::function<void()> callback) {
  request_pool_->set_callback4recv_msg(callback);
}

void RdmaNetwork::Read(const MemoryDescriptor& remote_memory_descriptor,
                       NetworkMemory* local_memory,
                       std::function<void()> callback) {
  Connection* conn =
      connection_pool_->GetConnection(remote_memory_descriptor.machine_id);
  CHECK(conn);

  Request* read_request = request_pool_->AllocRequest(true);
  CHECK(read_request);
  read_request->callback = callback;

  RdmaMemory* dst_memory = static_cast<RdmaMemory*>(local_memory);
  CHECK(dst_memory);

  conn->PostReadRequest(*read_request, remote_memory_descriptor, dst_memory);
}

bool RdmaNetwork::Poll(NetworkResult* result) {
  return PollRecvQueue(result) || PollSendQueue(result);
}

void RdmaNetwork::Barrier() {
  // Machine 0 acts as root. All machine send Barrier to root through the net
  // topology, and when root collects all Barrier, it replies the ReplyBarrier
  // to all machines through the net topology. The specific logic is,
  //
  // Machine i wait all its successor (neighbour whose id > i) machine's barrier
  // message. Only when all the Barrier message received, it send Barrier
  // message to its all predecessors (neighbours whose id < i) machines.
  // After Send Barrier Message OK, it then wait for the ReplyBarrier message
  // from the predecessors. When all ReplyBarrier received, it then send
  // ReplyBarrier to its all successor. After sending OK, the Barrier finish.

  int32_t num_predecessors = 0;
  int32_t num_successors = 0;
  for (auto peer_machine_id : net_topo_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id < my_machine_id_) ++num_predecessors;
    if (peer_machine_id > my_machine_id_) ++num_successors;
  }

  // 1. Wait for all the successor machines' barrier message
  NetworkResult result;
  for (int32_t i = 0; i < num_successors; ++i) {
    // Wait for the Barrier
    while (!PollRecvQueue(&result)) {
      CHECK(result.type == NetworkResultType::kReceiveMsg)
        << "Expected recv msg";
      CHECK(result.net_msg.type == NetworkMessageType::kBarrier)
        << "Expected MessageType::kBarrier";
    }
  }
  printf("Barrier wait all barrier over\n");

  // 2. Send to all the predecessors Barrier message
  NetworkMessage barrier_msg;
  barrier_msg.src_machine_id = my_machine_id_;
  barrier_msg.type = NetworkMessageType::kBarrier;
  for (auto peer_machine_id : net_topo_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id < my_machine_id_) {
      barrier_msg.dst_machine_id = peer_machine_id;
      SendMessage(barrier_msg);
      while (!PollSendQueue(&result))
        CHECK(result.type == NetworkResultType::kSendOk);
    }
  }
  printf("Barrier send all barrier over\n");

  // 3. Wait for the ReplyBarrier msg from predecessors
  // we shall poll 2 * n (num_predecessors) net event,
  // n for SEND_OK, n for RECEIVE_MSG
  for (int32_t i = 0; i < num_predecessors; ++i) {
    while (!PollRecvQueue(&result)) {
      CHECK(result.type == NetworkResultType::kReceiveMsg);
      CHECK(result.net_msg.type == NetworkMessageType::kReplyBarrier);
    }
  }
  printf("Barrier wait all reply barrier over\n");

  // 4. Send the Reply Barrier msg to all its
  NetworkMessage reply_barrier_msg;
  reply_barrier_msg.src_machine_id = my_machine_id_;
  reply_barrier_msg.type = NetworkMessageType::kReplyBarrier;
  for (auto peer_machine_id : net_topo_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id > my_machine_id_) {
      reply_barrier_msg.dst_machine_id = peer_machine_id;
      SendMessage(reply_barrier_msg);
      while (!PollSendQueue(&result)) {
        CHECK(result.type == NetworkResultType::kSendOk);
      }
    }
  }
  printf("Barrier send all reply barrier over\n");
}

void RdmaNetwork::InitConnections() {
  Connection* conn;
  for (auto peer_machine_id : net_topo_.all_nodes[my_machine_id_].neighbors) {
    conn = NewConnection();
    CHECK(conn);
    connection_pool_->AddConnection(peer_machine_id, conn);
  }
}

Connection* RdmaNetwork::NewConnection() {
  Connection* conn = new Connection(my_machine_id_);
  CHECK(conn);

  rdma_wrapper_->CreateConnector(conn);
  rdma_wrapper_->CreateQueuePair(conn);
  return conn;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::kReceiveMsg.
bool RdmaNetwork::PollRecvQueue(NetworkResult* result) {
  Request* request = rdma_wrapper_->PollRecvQueue(result);
  if (request == nullptr) { return false; }

  result->net_msg = request->rdma_msg->msg();
  request->callback();  // TODO(shiyuan)

  Connection* conn =
      connection_pool_->GetConnection(result->net_msg.src_machine_id);
  CHECK(conn);
  CHECK(request);
  conn->PostRecvRequest(*request);
  
  return true;
}

bool RdmaNetwork::PollSendQueue(NetworkResult* result) {
  Request* request = rdma_wrapper_->PollSendQueue(result);
  if (request == nullptr) { return false; }

  result->net_msg = request->rdma_msg->msg();
  request->callback();  // TODO(shiyuan)
  request_pool_->ReleaseRequest(request);
  
  return true;
}

const MemoryDescriptor& RdmaNetwork::GetMemoryDescriptor(
    int64_t register_id) const {
  auto mem_descriptor_it = register_id_to_mem_descriptor_.find(register_id);
  return mem_descriptor_it->second;
}

void RdmaNetwork::EstablishConnection() {
  // Connect to neighboring nodes with larger rank actively
  // For node with small rank, the connection will fail until its peer with
  // larger rank has established its own active connections
  Connection* conn = nullptr;
  Request* receive_request = nullptr;
  for (auto peer_machine_id : net_topo_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id > my_machine_id_) {
      receive_request = request_pool_->AllocRequest(false);
      CHECK(receive_request);
      do {
        conn = NewConnection();
        CHECK(conn);
        conn->Bind(net_topo_.all_nodes[my_machine_id_].address.c_str(), port_);
        conn->PostRecvRequest(*receive_request);
      } while (!conn->TryConnectTo(
          net_topo_.all_nodes[peer_machine_id].address.c_str(), port_));
      conn->CompleteConnectionTo();
      connection_pool_->AddConnection(peer_machine_id, conn);
      for (int k = 0; k < kPrePostRecvNumber; ++k) {
        receive_request = request_pool_->AllocRequest(false);
        CHECK(receive_request);
        conn->PostRecvRequest(*receive_request);
      }
    }
  }

  // Only if this node has established all the active connections, can it start
  // to listen and wait for the connections from peer nodes with smaller rank.
  for (auto peer_machine_id : net_topo_.all_nodes[my_machine_id_].neighbors) {
    // peer_machine_id means nothing here, just counting.
    if (peer_machine_id < my_machine_id_) {
      conn = NewConnection();
      CHECK(conn);
      receive_request = request_pool_->AllocRequest(false);
      CHECK(receive_request);
      // connecting with src_machine_id
      int64_t src_machine_id =
          rdma_wrapper_->WaitForConnection(conn, receive_request);
      CHECK_NE(src_machine_id, -1);
      connection_pool_->AddConnection(src_machine_id, conn);
      // Pre-post Receive issue before connect
      for (int k = 0; k < kPrePostRecvNumber; ++k) {
        receive_request = request_pool_->AllocRequest(false);
        CHECK(receive_request);
        conn->PostRecvRequest(*receive_request);
      }
    }
  }
}

}  // namespace oneflow
