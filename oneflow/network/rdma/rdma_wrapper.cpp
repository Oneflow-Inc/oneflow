#include "network/rdma/rdma_wrapper.h"

#include <ws2tcpip.h>  // TODO(shiyuan)
#include <vector>
// #include <iphlpapa.h>

#include "network/network.h"
#include "network/rdma/agency.h"
#include "network/rdma/connection_pool.h"
#include "network/rdma/message_pool.h"
#include "network/rdma/request_pool.h"

namespace oneflow {

namespace {
// TODO(shiyuan): Need move to network/windows。
//                Currently in Class rdma/windows and here.
//                Requires unified first parameter (machine_id and address).
sockaddr_in GetAddress(const uint64_t* machine_id, int port) {
  sockaddr_in addr = sockaddr_in();
  std::memset(&addr, 0, sizeof(sockaddr_in));
  // TODO(shiyuan) get address by machine_id.
  // inet_pton(AF_INET, address, &addr.sin_addr); // TODO(shiyuan)
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

RdmaWrapper::RdmaWrapper() {
  rdma_manager_ = NULL;
  request_pool_.reset(new RequestPool());
  connection_pool_.reset(new ConnectionPool());
  rdma_manager_ = new RdmaManager();
}

RdmaWrapper::~RdmaWrapper() {
  rdma_manager_->Destroy();
}

void RdmaWrapper::Init(uint64_t my_machine_id,
                       const NetworkTopology& net_topo) {
  my_machine_id_ = my_machine_id;
  net_topology_ = net_topo;

  InitConnections();

  // NdspiV2Open();
  // rdma_manager_->my_sock = GetAddress(my_machine_id);  // TODO(shiyuan)
  char* addr;
  // TODO(shiyuan) Get the address by my_machine_id_.
  rdma_manager_->Init(addr, port_);

  EstablishConnection();
}

void RdmaWrapper::Finalize() {
  register_id_to_mem_descriptor_.clear();
}

void RdmaWrapper::Barrier() {
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
  for (auto peer_machine_id : net_topology_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id < my_machine_id_)
      ++num_predecessors;
    if (peer_machine_id > my_machine_id_)
      ++num_successors;
  }

  // 1. Wait for all the successor machines' barrier message
  NetworkResult result;
  for (int32_t i = 0; i < num_successors; ++i) {
    // Wait for the Barrier
    while (!PollRecvQueue(&result))
      Sleep(1000);
    // CHECK(result.type == NetworkResultType::NET_RECEIVE_MSG)
    //   << "Expected recv msg";
    // CHECK(result.net_msg.type == NetworkMessageType::MSG_TYPE_BARRIER)
    //   << "Expected MSG_TYPE_BARRIER";
  }
  printf("Barrier wait all barrier over\n");

  // 2. Send to all the predecessors Barrier message
  NetworkMessage barrier_msg;
  barrier_msg.src_machine_id = my_machine_id_;
  barrier_msg.type = NetworkMessageType::MSG_TYPE_BARRIER;
  for (auto peer_machine_id : net_topology_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id < my_machine_id_) {
      barrier_msg.dst_machine_id = peer_machine_id;
      Send(barrier_msg);
      while (!PollSendQueue(&result))
        Sleep(1000);
      // CHECK(result.type == NetworkResultType::NET_SEND_OK);
    }
  }
  printf("Barrier send all barrier over\n");

  // 3. Wait for the ReplyBarrier msg from predecessors
  // we shall poll 2 * n (num_predecessors) net event,
  // n for SEND_OK, n for RECEIVE_MSG
  for (int32_t i = 0; i < num_predecessors; ++i) {
    while (!PollRecvQueue(&result))
      Sleep(1000);
    // CHECK(result.type == NetworkResultType::NET_RECEIVE_MSG);
    // CHECK(result.net_msg.type == NetworkMessageType::MSG_TYPE_REPLY_BARRIER);
  }
  printf("Barrier wait all reply barrier over\n");

  // 4. Send the Reply Barrier msg to all its
  NetworkMessage reply_barrier_msg;
  reply_barrier_msg.src_machine_id = my_machine_id_;
  reply_barrier_msg.type == NetworkMessageType::MSG_TYPE_REPLY_BARRIER;
  for (auto peer_machine_id : net_topology_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id > my_machine_id_) {
      reply_barrier_msg.dst_machine_id = peer_machine_id;
      Send(reply_barrier_msg);
      while (!PollSendQueue(&result))
        Sleep(1000);
      // CHECK(result.type == NetworkResultType::NET_SEND_OK);
    }
  }
  printf("Barrier send all reply barrier over\n");
}

NetworkMemory* RdmaWrapper::NewNetworkMemory() {
  return rdma_manager_->NewNetworkMemory();
}

// |msg| contains src machine_id and dst machine_id
bool RdmaWrapper::Send(const NetworkMessage& msg) {
  uint64_t dst_machine_id = msg.dst_machine_id;
  Connection* conn = connection_pool_->GetConnection(dst_machine_id);

  // 1. New network request, generating timestamp, get message memory
  //    from message pool.
  Request* send_request = request_pool_->AllocRequest(true);
  // check(send_request)

  // 2. Get network message buffer, copy the message.
  send_request->rdma_msg->mutable_msg() = msg;

  // NOTE(feiga): The last flag of Send, NO_OP_FLAG_SILENT_SUCCESS means the
  // successful sending will not generate an event in completion queue.
  // Change the flag to 0 if this does not suit for out design.
  // NOTE(jiyuan): We need to know the completion event of Send to recycle the
  // buffer of message.

  conn->PostSendRequest(send_request);
  // CHECK(!FAILED(result)) << "Failed to send\n";

  return true;  // TODO(shiyuan) return the result
}

void RdmaWrapper::Read(MemoryDescriptor* remote_memory_descriptor,
                       NetworkMemory* local_memory) {
  Connection* conn = connection_pool_->GetConnection(
      remote_memory_descriptor->machine_id);
  // CHECK

  Request* read_request = request_pool_->AllocRequest(true);
  // CHECK

  // Memorize the Request object for the most recently issued Read verb.
  time_stamp_of_last_read_request_ = read_request->time_stamp;
  // The read_request->registered_message->mutable_msg() will be set in
  // |RegisterEventMessage|.

  Memory* dst_memory = reinterpret_cast<Memory*>(local_memory);

  conn->PostReadRequest(read_request,
                               remote_memory_descriptor,
                               dst_memory);
  // CHECK
}

// TODO(shiyuan)
void RdmaWrapper::RegisterEventMessage(MsgPtr event_msg) {
  Request* last_read_request = request_pool_->GetRequest(
      time_stamp_of_last_read_request_);
  last_read_request->rdma_msg->mutable_msg().event_msg = (*event_msg);
}

bool RdmaWrapper::Poll(NetworkResult* result) {
  return PollRecvQueue(result) || PollSendQueue(result);
}

void RdmaWrapper::InitConnections() {
  Connection* conn;
  for (auto peer_machine_id : net_topology_.all_nodes[my_machine_id_].neighbors) {
    conn = NewConnection();
    connection_pool_->AddConnection(peer_machine_id, conn);
  }
}

Connection* RdmaWrapper::NewConnection() {
  Connection* conn = new Connection();

  rdma_manager_->CreateConnector(conn);
  rdma_manager_->CreateQueuePair(conn);

  return conn;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::NET_RECEIVE_MSG.
bool RdmaWrapper::PollRecvQueue(NetworkResult* result) {
  int32_t time_stamp = rdma_manager_->PollRecvQueue(result);
  if (time_stamp == -1)
    return false;
  Request* request = request_pool_->GetRequest(time_stamp);
  // CHECK request

  result->net_msg = request->rdma_msg->msg();

  // Equivalent to:
  //  request_pool_->ReleaseRequest(time_stamp);
  //  PostReceiveRequest(result->net_msg.src_rank);
  // but is more efficient
  RePostRecvRequest(result->net_msg.src_machine_id, time_stamp);
  return true;
}

bool RdmaWrapper::PollSendQueue(NetworkResult* result) {
  int32_t time_stamp = rdma_manager_->PollSendQueue(result);
  if (time_stamp == -1)
    return false;
  Request* request = request_pool_->GetRequest(time_stamp);

  result->net_msg = request->rdma_msg->msg();
  request_pool_->ReleaseRequest(time_stamp);
  return true;
}

void RdmaWrapper::RePostRecvRequest(uint64_t peer_machine_id,
                                    int32_t time_stamp) {
  Connection* conn = connection_pool_->GetConnection(peer_machine_id);

  Request* receive_request = request_pool_->UpdateTimeStampAndReuse(time_stamp);

  conn->PostRecvRequest(receive_request);
}

const MemoryDescriptor& RdmaWrapper::GetMemoryDescriptor(
    int64_t register_id) const {
  auto mem_descriptor_it = register_id_to_mem_descriptor_.find(register_id);
  // CHECK
  return mem_descriptor_it->second;
}

void RdmaWrapper::EstablishConnection() {
  // Connect to neighboring nodes with larger rank actively
  // For node with small rank, the connection will fail until its peer with
  // larger rank has established its own active connections
  Connection* conn = NULL;
  Request* receive_request = NULL;
  for (auto peer_machine_id : net_topology_.all_nodes[my_machine_id_].neighbors) {
    if (peer_machine_id > my_machine_id_) {
      conn = connection_pool_->GetConnection(peer_machine_id);
      conn->Bind();
      receive_request = request_pool_->AllocRequest(false);
      conn->PostRecvRequest(receive_request);
      while (conn->TryConnectTo()) {
        // If connection failed, wait and retry again.
        Sleep(2000);
      }
      conn->CompleteConnectionTo();

      for (int k = 0; k < kPrePostRecvNumber; ++k) {
        receive_request = request_pool_->AllocRequest(false);
        conn->PostRecvRequest(receive_request);
      }
    }
  }

  // Only if this node has established all the active connections, can it start
  // to listen and wait for the connections from peer nodes with smaller rank.
  Connection* temp_conn = NULL;
  temp_conn = NewConnection();
  for (auto peer_machine_id : net_topology_.all_nodes[my_machine_id_].neighbors) {
    // peer_machine_id means nothing here, just counting.
    if (peer_machine_id < my_machine_id_) {
      // connecting with src_machine_id
      uint64_t src_machine_id = rdma_manager_->WaitForConnection(temp_conn);
      conn = connection_pool_->GetConnection(src_machine_id);
      // Pre-post Receive issue before connect
      receive_request = request_pool_->AllocRequest(false);
      conn->PostRecvRequest(receive_request);
      conn->AcceptConnect();
      for (int k = 0; k < kPrePostRecvNumber; ++k) {
        receive_request = request_pool_->AllocRequest(false);
        conn->PostRecvRequest(receive_request);
      }
    }
  }
}

}  // namespace oneflow
