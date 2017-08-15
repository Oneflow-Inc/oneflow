#include "oneflow/core/network/rdma/windows/connection.h"
#include <WS2tcpip.h>
#include <ndspi.h>
#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const std::string& ip, int32_t port) {
  sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

Connection::Connection(int64_t my_machine_id)
    : my_machine_id_(my_machine_id),
      connector_(nullptr),
      queue_pair_(nullptr),
      ov_(new OVERLAPPED) {
  ov_->hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
}

Connection::~Connection() { Destroy(); }

void Connection::Bind(const std::string& my_ip, int32_t my_port) {
  sockaddr_in my_sock = GetAddress(my_ip, my_port);
  HRESULT hr = connector_->Bind(reinterpret_cast<const sockaddr*>(&my_sock),
                                sizeof(my_sock));
  CHECK(!FAILED(hr)) << "Connector bind failed";
}

bool Connection::TryConnectTo(const std::string& peer_ip, int32_t peer_port) {
  sockaddr_in peer_addr = GetAddress(peer_ip, peer_port);
  HRESULT hr = connector_->Connect(
      queue_pair_, reinterpret_cast<const sockaddr*>(&peer_addr),
      sizeof(peer_addr),
      10,               // inbound read limit, max in-flight number
      10,               // outbound read limit, max in-flight number
      &my_machine_id_,  // Send the active side machine id as private data to
                        // tell the passive side who is the sender.
      sizeof(int64_t), ov_);

  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, TRUE);
  }

  if (SUCCEEDED(hr)) {
    return true;
  } else {
    return false;
  }
}

void Connection::CompleteConnection() {
  HRESULT hr;
  hr = connector_->CompleteConnect(ov_);
  if (hr == ND_PENDING) { hr = connector_->GetOverlappedResult(ov_, TRUE); }
  CHECK(SUCCEEDED(hr)) << "CompleteConnect failed";
}

void Connection::AcceptConnect() {
  HRESULT hr =
      connector_->Accept(queue_pair_,
                         10,       // inbound limit
                         10,       // outbound limit
                         nullptr,  // add private data? // Credit information?
                         0, ov_);
  if (hr == ND_PENDING) { hr = connector_->GetOverlappedResult(ov_, TRUE); }
  CHECK(SUCCEEDED(hr)) << "Fail accept connection";
}

void Connection::Destroy() {
  if (connector_ != nullptr) {
    HRESULT hr = connector_->Disconnect(ov_);
    if (hr == ND_PENDING) { hr = connector_->GetOverlappedResult(ov_, TRUE); }
    connector_->Release();
  }
  delete ov_;
  ov_ = nullptr;
  if (queue_pair_ != nullptr) { queue_pair_->Release(); }
}

void Connection::PostSendRequest(const Request& send_request) {
  HRESULT hr = queue_pair_->Send(
      (void*)&send_request,
      static_cast<const ND2_SGE*>(send_request.rdma_msg->net_memory()->sge()),
      1, 0);
  CHECK(SUCCEEDED(hr));
}

void Connection::PostRecvRequest(const Request& recv_request) {
  HRESULT hr = queue_pair_->Receive(
      (void*)&recv_request,
      static_cast<const ND2_SGE*>(recv_request.rdma_msg->net_memory()->sge()),
      1);
  CHECK(SUCCEEDED(hr));
}

void Connection::PostReadRequest(
    const Request& read_request,
    const MemoryDescriptor& remote_memory_descriptor, RdmaMemory* dst_memory) {
  HRESULT hr = queue_pair_->Read(
      (void*)&read_request, static_cast<const ND2_SGE*>(dst_memory->sge()), 1,
      remote_memory_descriptor.address, remote_memory_descriptor.remote_token,
      0);  // TODO(shiyuan) parameters
  CHECK(SUCCEEDED(hr));
}

}  // namespace oneflow
