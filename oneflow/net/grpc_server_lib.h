#ifndef GRPC_SERVER_H_
#define GRPC_SERVER_H_

#include "grpc++/grpc++.h"
#include "async_service_interface.h"
#include "master_env.h"
#include "worker_env.h"
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

  MasterEnv master_env_;
  WorkerEnv worker_env_;
  std::unique_ptr<::grpc::Server> server_;
};

}//end namespace oneflow

#endif


