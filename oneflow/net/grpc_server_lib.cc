#include <iostream>
#include <memory>
#include <string>
#include <grpc++/grpc++.h>

#include "master_env.h"
#include "worker_env.h"
#include "grpc_worker_cache.h"
#include "grpc_channel.h"
#include "grpc_server_lib.h"
#include "grpc_master_service.h"
//#include "grpc_worker_service.h"
#include "master_session.h"

namespace oneflow{

GrpcServer::GrpcServer(){}
GrpcServer::~GrpcServer(){
  delete master_service_;
  //delete worker_service_;
}

int GrpcServer::Init(){
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:50051", ::grpc::InsecureServerCredentials());
  //master service and worker service use the same builder
  master_service_ = NewGrpcMasterService(&builder);
  //worker_service_ = NewGrpcWorkerService(&builder);
  server_ = builder.BuildAndStart();
  //master servie and woker service use the same channle_cache
  std::unique_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(GetChannelCreationFunction()));
  worker_env_.worker_cache = NewGrpcWorkerCache(channel_cache.release());
  master_env_.worker_cache = worker_env_.worker_cache;

  master_env_.master_session_factory = NewMasterSession;
}

int GrpcServer::Start(){
  master_thread_.reset(
      env_->StartThread(ThreadOptions(), "master_service", [this] {master_service_->HandleRPCsLoop();}));
  //worker_thread_.reset(
  //    env_->StartThread(ThreadOptions(), "worker_service", [this] {worker_service_->HandleRPCsLoop();})); 
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  return NewHostPortGrpcChannel;
}

void GrpcServer::NewServer(){
  std::unique_ptr<GrpcServer> ret(new GrpcServer());
  ret->Init();
}

}
