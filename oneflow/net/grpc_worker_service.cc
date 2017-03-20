#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>
#include "async_service_interface.h"
#include "grpc_worker_service.h"
#include "oneflow.pb.h"
#include "oneflow.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using of::Role;
using of::Rank;
using of::Status_all;
using of::comm;

namespace oneflow{

class GrpcWorkerService : public AsyncServiceInterface{
 public:
  GrpcWorkerService(::grpc::ServerBuilder* builder){
    builder->RegisterService(&worker_service_);  
    cq_ = builder->AddCompletionQueue().release();
  }

  void HandleRPCsLoop() {}

  ::grpc::ServerCompletionQueue* cq_;
  of::comm::AsyncService worker_service_;
};

AsyncServiceInterface* NewGrpcWorkerService(::grpc::ServerBuilder* builder){
  return new GrpcWorkerService(builder);
}

}
