#ifndef GRPC_SERVER_H_
#define GRPC_SERVER_H_

#include "grpc++/grpc++.h"
#include "async_service_interface.h"
#include "master_env.h"
#include "worker_env.h"
#include "grpc_channel.h"
#include "platform_env.h"
#include "server_lib.h"

namespace oneflow{

class GrpcServer : public ServerInterface{
 protected:
  GrpcServer();
 public:
  //GrpcServer();
  virtual ~GrpcServer();
  
  void Init();
  void Start() override;
  void Join() override;
  virtual ChannelCreationFunction GetChannelCreationFunction() const;
  static void Create(std::unique_ptr<ServerInterface>* out_server);
  AsyncServiceInterface* master_service_ = nullptr;
  AsyncServiceInterface* worker_service_ = nullptr;

  Env* env_;
  MasterEnv master_env_;
  WorkerEnv worker_env_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<Thread> master_thread_;
  //std::unique_ptr<Thread> worker_thread_;
};

}//end namespace oneflow

#endif


