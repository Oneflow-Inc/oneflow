#include "network/grpc/grpc_worker.h"

namespace oneflow {

GrpcWorker::GrpcWorker() {}
GrpcWorker::~GrpcWorker() {}


bool GrpcWorker::Send(const NetworkMessage& msg) {

  return true;
}

void GrpcWorker::Read(MemoryDescriptor* src, NetworkMemory* dst) {

}

}



