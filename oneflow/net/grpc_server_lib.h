#ifndef GRPC_SERVER_H_
#define GRPC_SERVER_H_

#include "grpc++/grpc++.h"
#include "async_service_interface.h"
#include "master_env.h"
#include "worker_env.h"
#include "grpc_channel.h"
#include "platform_env.h"

namespace oneflow{

class GrpcServer{
 public:
  GrpcServer();
  ~GrpcServer();
  
  int Init();
  int Start();
  virtual ChannelCreationFunction GetChannelCreationFunction() const;
  void NewServer();
  AsyncServiceInterface* master_service_ = nullptr;
  AsyncServiceInterface* worker_service_ = nullptr;

  Env* env_;
  MasterEnv master_env_;
  WorkerEnv worker_env_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<Thread> master_threads_;
  std::unique_ptr<Thread> worker_threads_;
};

}//end namespace oneflow

#endif


