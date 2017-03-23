#include <iostream>
#include <memory>
#include <string>

//#include "grpc_worker_service_impl.h"

#include <grpc++/grpc++.h>
#include "grpc++/server_builder.h"

#include "async_service_interface.h"
#include "grpc_worker_service.h"
#include "grpc_worker_service_impl.h"
#include "worker.pb.h"


namespace oneflow{

class GrpcWorkerService : public AsyncServiceInterface{
 public:
  GrpcWorkerService(::grpc::ServerBuilder* builder){
    builder->RegisterService(&worker_service_);  
    cq_ = builder->AddCompletionQueue().release();
  }

  void HandleRPCsLoop() {}

  ::grpc::ServerCompletionQueue* cq_;
  grpc::WorkerService::AsyncService worker_service_;
};

AsyncServiceInterface* NewGrpcWorkerService(::grpc::ServerBuilder* builder){
  return new GrpcWorkerService(builder);
}

}
