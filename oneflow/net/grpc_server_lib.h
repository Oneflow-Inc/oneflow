#ifndef GRPC_SERVER_H_
#define GRPC_SERVER_H_

#include "grpc++/grpc++.h"
#include "async_service_interface.h"
#include "grpc_channel.h"

namespace oneflow{

class GrpcServer{
 public:
  GrpcServer();
  ~GrpcServer();
  
  int Init();
  virtual ChannelCreationFunction GetChannelCreationFunction() const;
  void NewServer();
  AsyncServiceInterface* master_service_ = nullptr;
  AsyncServiceInterface* worker_service_ = nullptr;

  std::unique_ptr<::grpc::Server> server_;
};

}//end namespace oneflow

#endif


