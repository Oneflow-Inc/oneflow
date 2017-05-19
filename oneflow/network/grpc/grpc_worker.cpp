#include "network/grpc/grpc_worker.h"

namespace oneflow {

GrpcWorker::GrpcWorker() {}
GrpcWorker::~GrpcWorker() {}


bool GrpcWorker::Send(const NetworkMessage& msg) {
  //oneflow::NetworkMessageRpc should filled by msg
  oneflow:EventMessageRpc event_message;
  event_message.set_envent_message_type(0);
  
  oneflow::NetworkMessageRpc network_message;
  network_message.set_network_message_type(0);   
  network_message.set_allocated_event_message(&event_message);

  oneflow::SendMessageRequest req;
  req.set_allocated_network_message(&network_message);

  return true;
}

void GrpcWorker::Read(MemoryDescriptor* src, NetworkMemory* dst) {

}

}



