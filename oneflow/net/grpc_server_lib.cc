#include <iostream>
#include <memory>
#include <string>
#include <grpc++/grpc++.h>

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
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  return NewHostPortGrpcChannel;
}

void GrpcServer::NewServer(){
  std::unique_ptr<GrpcServer> ret(new GrpcServer());
  ret->Init();
}

}
