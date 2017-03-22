#include <iostream>
#include <memory>
#include <string>
#include <grpc++/grpc++.h>

#include "master_env.h"
#include "worker_env.h"
#include "grpc_worker_cache.h"
#include "grpc_channel.h"
#include "server_lib.h"
#include "grpc_server_lib.h"
#include "grpc_master_service.h"
#include "async_service_interface.h"
//#include "grpc_worker_service.h"
#include "master_session.h"
#include "platform_env.h"

namespace oneflow{

GrpcServer::GrpcServer(){}
GrpcServer::~GrpcServer(){
  //delete master_service_;
  //delete worker_service_;
}

void GrpcServer::Init(){
  std::cout<<"Init start------"<<std::endl;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:50051", ::grpc::InsecureServerCredentials());
  //master service and worker service use the same builder
  if(master_service_ == nullptr) std::cout<<"before nullptr---"<<std::endl;
  master_service_ = NewGrpcMasterService(&builder);
  if(master_service_ == nullptr) std::cout<<"after nullptr---"<<std::endl;
  //worker_service_ = NewGrpcWorkerService(&builder);
  server_ = builder.BuildAndStart();
  //master servie and woker service use the same channle_cache
  std::unique_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(GetChannelCreationFunction()));
  worker_env_.worker_cache = NewGrpcWorkerCache(channel_cache.release());
  master_env_.worker_cache = worker_env_.worker_cache;

  master_env_.master_session_factory = NewMasterSession;
  std::cout<<"Init end------"<<std::endl;
}

void GrpcServer::Start(){
  //if(master_service_ == nullptr) {std::cout<<"nullprt"<<std::endl;}
  //master_service_->HandleRPCsLoop();
  //master_thread_.reset();
  std::cout<<"start master service------"<<std::endl;
  master_thread_.reset(
      //StartThread(ThreadOptions(), "master_service", [this] {std::cout<<"hi"<<std::endl;});
      StartThread(ThreadOptions(), "master_service", [this] {master_service_->HandleRPCsLoop();}));
  //worker_thread_.reset(
  //    env_->StartThread(ThreadOptions(), "worker_service", [this] {worker_service_->HandleRPCsLoop();})); 
}

void GrpcServer::Join() {}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  return NewHostPortGrpcChannel;
}

void GrpcServer::Create(std::unique_ptr<ServerInterface>* out_server) {
  std::cout<<"create GrpcServer"<<std::endl;
  std::unique_ptr<GrpcServer> ret(new GrpcServer());
  ret->Init();
  *out_server = std::move(ret);
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  void NewServer(std::unique_ptr<ServerInterface>* out_server) override {
    std::cout<<"new server"<<std::endl;
    GrpcServer::Create(out_server);
  }
};
}

}
