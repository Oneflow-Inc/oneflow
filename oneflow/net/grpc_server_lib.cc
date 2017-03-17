#include <iostream>
#include <memory>
#include <string>
#include <grpc++/grpc++.h>

#include "grpc_server_lib.h"
#include "master_service.h"
#include "worker_service.h"

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
}

void GrpcServer::NewServer(){
  std::unique_ptr<GrpcServer> ret(new GrpcServer());
  ret->Init();
}

}
