#include "oneflow/core/network/rdma/windows/connection.h"
#include <ndspi.h>
#include <iostream>
#include "oneflow/core/network/rdma/windows/interface.h"
#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

namespace {

sockaddr_in GetSocket(const char* address, int port) {
  sockaddr_in sock = sockaddr_in();
  memset(&sock, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, address, &sock.sin_addr);
  sock.sin_family = AF_INET;
  sock.sin_port = htons(static_cast<u_short>(port));
  return sock;
}

}  // namespace

Connection::Connection(uint64_t my_machine_id)
    : Connection::Connection(my_machine_id, -1) {}

Connection::Connection(uint64_t my_machine_id, uint64_t peer_machine_id) {
  my_machine_id_ = my_machine_id;
  peer_machine_id_ = peer_machine_id;
  connector_ = NULL;
  queue_pair_ = NULL;
  // TODO(shiyuan)
  ov_ = new OVERLAPPED;
  ov_->hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
}

Connection::~Connection() {
}

bool Connection::Bind(const char* my_address, int port) {
  sockaddr_in my_sock = GetSocket(my_address, port);
  HRESULT hr = connector_->Bind(reinterpret_cast<const sockaddr*>(&my_sock),
                               sizeof(my_sock));
  if (SUCCEEDED(hr)) {
    return true;
  } else {
    return false;
  }
}

bool Connection::TryConnectTo(const char* peer_address, int port) {
  sockaddr_in peer_sock = GetSocket(peer_address, port);
  HRESULT hr = connector_->Connect(
      queue_pair_,
      reinterpret_cast<const sockaddr*>(&peer_sock),
      sizeof(peer_sock),
      10,      // inbound read limit, max in-flight number
      10,      // outbound read limit, max in-flight number
      &my_machine_id_,  // Send the active side machine id as private data to
                        // tell the passive side who is the sender.
      sizeof(uint64_t),
      ov_);

  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, TRUE);
    std::cout << "ND_PENDING" << std::endl;
  }

  if (SUCCEEDED(hr)) {
    return true;
  } else {
    std::cout << "Failed try to connect" << std::endl;
    return false;
  }
}

void Connection::CompleteConnectionTo() {
  HRESULT hr;
  hr = connector_->CompleteConnect(ov_);
  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, TRUE);
  }
  if (FAILED(hr) || hr == ND_TIMEOUT) {
    std::cout << "CompleteConnect failed" << std::endl;
  }
  // CHECK(!FAILED(hr)) << "Failed to complete connection\n";
}

void Connection::AcceptConnect() {
  HRESULT hr = connector_->Accept(queue_pair_,
      10,  // inbound limit
      10,  // outbound limit
      NULL,  // add private data? // Credit information?
      0,
      ov_);
  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(ov_, true);
  }
  if (SUCCEEDED(hr)) {
    std::cout << "Success accept connection" << std::endl;
  } else {
    std::cout << "Fail accept connection" << std::endl;
  }
  // CHECK(!FAILED(hr)) << "Failed to accept\n";
  // LOG(INFO) << "Accept done\n";
}

void Connection::DestroyConnection() {
}

void Connection::PostSendRequest(Request* send_request) {
  queue_pair_->Send(
      &send_request->time_stamp,
      static_cast<const ND2_SGE*>(
          send_request->rdma_msg->net_memory()->sge()),
      1,
      0);  // TODO(shiyuan) this flag should be mod for generate an event in cq
}

void Connection::PostRecvRequest(Request* recv_request) {
  HRESULT hr = queue_pair_->Receive(
      &recv_request->time_stamp,
      static_cast<const ND2_SGE*>(
          recv_request->rdma_msg->net_memory()->sge()),
      1);
}

void Connection::PostReadRequest(
    Request* read_request,
    MemoryDescriptor* remote_memory_descriptor,
    RdmaMemory* dst_memory) {
  HRESULT hr = queue_pair_->Read(
      &read_request->time_stamp,
      static_cast<const ND2_SGE*>(dst_memory->sge()),
      1,
      remote_memory_descriptor->address,
      remote_memory_descriptor->remote_token,
      0);  // TODO(shiyuan) parameters
}

}  // namespace oneflow
