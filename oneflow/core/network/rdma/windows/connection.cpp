#include "oneflow/core/network/rdma/windows/connection.h"
#include <ndspi.h>
#include "oneflow/core/network/rdma/windows/interface.h"
#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const char* ip, int32_t port) {
  sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, address, &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

Connection::Connection(int64_t my_machine_id)
    : Connection::Connection(my_machine_id, -1) {}

Connection::Connection(int64_t my_machine_id, int64_t peer_machine_id)
    : connector_(nullptr),
      queue_pair_(nullptr),
      ov_(new OVERLAPPED),
      my_machine_id_(my_machine_id),
      peer_machine_id_(peer_machine_id) {
  ov_->hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
}

Connection::~Connection() {
  DestroyConnection();
}

void Connection::set_connector(IND2Connector* connector) {
  CHECK(!connector_);
  connector_ = connector;
}

void Connection::set_queue_pair(IND2QueuePair* queue_pair) {
  CHECK(!queue_pair_);
  queue_pair_ = queue_pair;
}

void Connection::set_overlapped(OVERLAPPED* ov) {
  CHECK(!ov_);
  ov_ = ov;
}

void Connection::Bind(const char* my_ip, int32_t my_port) {
  sockaddr_in my_addr = GetAddress(my_ip, my_port);
  HRESULT hr = connector_->Bind(reinterpret_cast<const sockaddr*>(&my_addr),
                                sizeof(my_addr));
  CHECK(SUCCEEDED(hr));
}

bool Connection::TryConnectTo(const char* peer_ip, int32_t peer_port) {
  sockaddr_in peer_addr = GetAddress(peer_ip, peer_port);
  CHECK(peer_addr);
  HRESULT hr = connector_->Connect(
      queue_pair_,
      reinterpret_cast<const sockaddr*>(&peer_addr),
      sizeof(peer_addr),
      10,      // inbound read limit, max in-flight number
      10,      // outbound read limit, max in-flight number
      &my_machine_id_,  // Send the active side machine id as private data to
                        // tell the passive side who is the sender.
      sizeof(int64_t),
      ov_);

  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, TRUE);
    std::cout << "ND_PENDING" << std::endl;
  }

  if (SUCCEEDED(hr)) {
    return true;
  } else {
    return false;
  }
}

void Connection::CompleteConnectionTo() {
  HRESULT hr;
  hr = connector_->CompleteConnect(ov_);
  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, TRUE);
  }
  CHECK(SUCCEEDED(hr))<< "CompleteConnect failed";
}

void Connection::AcceptConnect() {
  HRESULT hr = connector_->Accept(queue_pair_,
      10,  // inbound limit
      10,  // outbound limit
      nullptr,  // add private data? // Credit information?
      0,
      ov_);
  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, true);
  }
  CHECK(SUCCEEDED(hr)) << "Fail accept connection";
}

void Connection::DestroyConnection() {
  if (connector_ != nullptr) {
    HRESULT hr = connector_->Disconnect(ov_);
    if (hr == ND_PENDING) {
      SIZE_T BytesRet;
      hr = connector_->GetOverlappedResult(ov_, &BytesRet, TRUE);
    }
    connector_->Release();
  }
  delete ov_;
  ov_ = nullptr;
  if (queue_pair_ != nullptr) {
    queue_pair_->Release();    
  }
}

void Connection::PostSendRequest(const Request& send_request) {
  HRESULT hr = queue_pair_->Send(
      &send_request,
      static_cast<const ND2_SGE*>(
          send_request.rdma_msg->net_memory()->sge()),
      1,
      0);
  CHECK(SUCCEEDED(hr));
}

void Connection::PostRecvRequest(const Request& recv_request) {
  HRESULT hr = queue_pair_->Receive(
      &recv_request,
      static_cast<const ND2_SGE*>(
          recv_request.rdma_msg->net_memory()->sge()),
      1);
  CHECK(SUCCEEDED(hr));
}

void Connection::PostReadRequest(
    const Request& read_request,
    const MemoryDescriptor& remote_memory_descriptor,
    RdmaMemory* dst_memory) {
  HRESULT hr = queue_pair_->Read(
      &read_request,
      static_cast<const ND2_SGE*>(dst_memory->sge()),
      1,
      remote_memory_descriptor.address,
      remote_memory_descriptor.remote_token,
      0);  // TODO(shiyuan) parameters
  CHECK(SUCCEEDED(hr));
}

}  // namespace oneflow
