#ifndef GRPC_WORKER_H
#define GRPC_WORKER_H

#include "distributed_runtime/worker.pb.h"
#include "distributed_runtime/worker_service.pb.h"
#include "network/network.h"
#include "distributed_runtime/grpc_remote_worker.h"

namespace oneflow {

struct NetworkMessage;
struct MemoryDescriptor;
struct NetworkMemory;

class GrpcWorker : public Network {
  public:
    GrpcWorker();
    ~GrpcWorker();
    bool Send(const NetworkMessage& msg);
    void Read(MemoryDescriptor* src, NetworkMemory* dst);

    GrpcRemoteWorker* remote_worker;
};

}

#endif


