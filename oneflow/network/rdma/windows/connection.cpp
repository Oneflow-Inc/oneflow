#include "connection.h"

namespace oneflow {

Connection::Connection() : Connection::Connection(-1) {}

Connection::Connection(uint64_t peer_machine_id)
{
  peer_machine_id_ = peer_machine_id;

  connector_ = NULL;
  queue_pair_ = NULL;
  // recv_region_; // GetMemory

  // TODO(shiyuan) 
  ov_.hEvent = CreateEvent(NULL, false, false, NULL);

}

Connection::~Connection()
{

}

bool Connection::Bind() {
  // TODO(shiyuan) add init of my_sock and peer_sock in Connection::Init
  return connector_->Bind(reinterpret_cast<const sockaddr*>(&my_sock), 
      sizeof(my_sock));
  // CHECK(!FAILED(hr)) << "Connector bind failed.\n"
} 

bool Connection::TryConnectTo()
{
  HRESULT hr = connector_->Connect(
      queue_pair_,
      reinterpret_cast<const sockaddr*>(&peer_sock),
      sizeof(peer_sock),
      0,         // TODO(shiyuan): 0 indicate we do not support Read, right here?
      0,         // TODO(shiyuan): 0 indicate we do not support Read, right here?
      &my_machine_id_,  // Send the active side machine id as private data to tell
                        // the passive side who is the sender. 
      sizeof(uint64_t), // is this equal to size of my_machine_id_?
      &ov);
  
  if (hr == ND_PENDING) {
    hr = connector_->GetOverlappedResult(&ov, TRUE);
  }

  if (hr != ND_SUCCESS) {
    // hr = connector_->CleanConnection(peer_rank);
    return false;
  }

  return true;
}

void Connection::CompleteConnectionTo()
{
  connector_->CompleteConnect(&ov);
  if (hr == ND_PENDIND) {
    hr = connector_->GetOverlappedResult(&ov, TRUE);
  }
  // CHECK(!FAILED(hr)) << "Failed to complete connection\n";
}

void Connection::AcceptConnect()
{
  HRESULT hr = connector->Accept(queue_pair,
      0, // Zero, we don't allow Read. ??
      0, // Zero, we don't allow Read. ??
      NULL, // TODO(feiga): add private data? // Credit information?
      0, 
      &ov);
  if (hr == ND_PENDING) {
    hr = connector->GetOverlappedResult(&ov, true);
  }
  // CHECK(!FAILED(hr)) << "Failed to accept\n";
  // LOG(INFO) << "Accept done\n";
}

void Connection::PostToRecvRequestQueue(Request* receive_request) // TODO(shiyuan) parameter pass
{
  queue_pair_->Receive(
      &receive_request->time_stamp,
      static_cast<const ND2_SGE*> (
          receive_request->registered_message->net_memory()->sge()),
      1);
}

void DestroyConnection()
{

}

} // namespace oneflow

