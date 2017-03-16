#ifndef GRPC_SERVER_H_
#define GRPC_SERVER_H_

#include "grpc++/grpc++.h"
#include "async_service_interface.h"

namespace oneflow{

class GrpcServer{
 public:
  GrpcServer();
  ~GrpcServer();
  
  int Init();
  void NewServer();
  AsyncServiceInterface* master_service_ = nullptr;
  AsyncServiceInterface* worker_service_ = nullptr;

  std::unique_ptr<::grpc::Server> server_;
};

}//end namespace oneflow

#endif


