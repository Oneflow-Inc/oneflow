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
#include "grpc_worker_service.h"

namespace oneflow{

GrpcServer::GrpcServer(){}
GrpcServer::~GrpcServer(){
  delete master_service_;
  delete worker_service_;
}

int GrpcServer::Init(){
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:50051", ::grpc::InsecureServerCredentials());
  master_service_ = NewGrpcMasterService(&builder);
  worker_service_ = NewGrpcWorkerService(&builder);
  server_ = builder.BuildAndStart();

  std::unique_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(GetChannelCreationFunction()));
  worker_env_.worker_cache = NewGrpcWorkerCache(channel_cache.release());
  master_env_.worker_cache = worker_env_.worker_cache;
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  return NewHostPortGrpcChannel;
}

void GrpcServer::NewServer(){
  std::unique_ptr<GrpcServer> ret(new GrpcServer());
  ret->Init();
}

}
