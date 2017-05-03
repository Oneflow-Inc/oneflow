#include "network/rdma/windows/connection.h"
#include "ndspi.h"  // TODO(shiyuan)
#include "network/rdma/windows/interface.h"

namespace oneflow {

Connection::Connection() : Connection::Connection(-1) {}

Connection::Connection(uint64_t peer_machine_id) {
  peer_machine_id_ = peer_machine_id;

  connector = NULL;
  queue_pair = NULL;
  // recv_region_; // GetMemory

  // TODO(shiyuan)
  ov.hEvent = CreateEvent(NULL, false, false, NULL);
}

Connection::~Connection() {
}

bool Connection::Bind() {
  // TODO(shiyuan) add init of my_sock and peer_sock in Connection::Init
  return connector->Bind(reinterpret_cast<const sockaddr*>(&my_sock_),
      sizeof(my_sock_));
  // CHECK(!FAILED(hr)) << "Connector bind failed.\n"
}

bool Connection::TryConnectTo() {
  HRESULT hr = connector->Connect(
      queue_pair,
      reinterpret_cast<const sockaddr*>(&peer_sock_),
      sizeof(peer_sock_),
      0,      // TODO(shiyuan): 0 indicate we do not support Read, right here?
      0,      // TODO(shiyuan): 0 indicate we do not support Read, right here?
      &my_machine_id_,  // Send the active side machine id as private data to tell
                        // the passive side who is the sender.
      sizeof(uint64_t),  // is this equal to size of my_machine_id_?
      &ov);

  if (hr == ND_PENDING) {
    hr = connector->GetOverlappedResult(&ov, true);
  }

  if (hr != ND_SUCCESS) {
    // hr = connector_->CleanConnection(peer_rank);
    return false;
  }

  return true;
}

void Connection::CompleteConnectionTo() {
  HRESULT hr;
  connector->CompleteConnect(&ov);
  if (hr == ND_PENDING) {
    hr = connector->GetOverlappedResult(&ov, true);
  }
  // CHECK(!FAILED(hr)) << "Failed to complete connection\n";
}

void Connection::AcceptConnect() {
  HRESULT hr = connector->Accept(queue_pair,
      0,  // Zero, we don't allow Read. ??
      0,  // Zero, we don't allow Read. ??
      NULL,  // TODO(feiga): add private data? // Credit information?
      0,
      &ov);
  if (hr == ND_PENDING) {
    hr = connector->GetOverlappedResult(&ov, true);
  }
  // CHECK(!FAILED(hr)) << "Failed to accept\n";
  // LOG(INFO) << "Accept done\n";
}
void Connection::PostToSendRequestQueue(Request* send_request) {
  queue_pair->Send(
      &send_request->time_stamp,
      static_cast<const ND2_SGE*>(
          send_request->rdma_msg->net_memory()->sge()),
      1,
      0);  // TODO(shiyuan) this flag should be mod for generate an event in cq
}

void Connection::PostToRecvRequestQueue(Request* recv_request) {
  queue_pair->Receive(
      &recv_request->time_stamp,
      static_cast<const ND2_SGE*>(
          recv_request->rdma_msg->net_memory()->sge()),
      1);
}

void Connection::PostToReadRequestQueue(
    Request* read_request,
    MemoryDescriptor* remote_memory_descriptor,
    Memory* dst_memory) {
  queue_pair->Read(
      &read_request->time_stamp,
      static_cast<const ND2_SGE*>(dst_memory->sge()),
      1,
      remote_memory_descriptor->address,
      remote_memory_descriptor->remote_token,
      0);  // TODO(shiyuan) parameters
}

void DestroyConnection() {
}

}  // namespace oneflow
