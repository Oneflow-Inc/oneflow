#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>
#include "async_service_interface.h"
#include "grpc_master_service.h"
#include "oneflow.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using of::Role;
using of::Rank;
using of::Status_all;
using of::comm;

namespace oneflow{

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(::grpc::ServerBuilder* builder){
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue().release();
  }

 private:
  ::grpc::ServerCompletionQueue* cq_;
  of::comm::AsyncService master_service_;
};

AsyncServiceInterface* NewGrpcMasterService(::grpc::ServerBuilder* builder){
  return new GrpcMasterService(builder);
}

}
