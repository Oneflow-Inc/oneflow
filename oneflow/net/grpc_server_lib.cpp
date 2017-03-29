#include <iostream>
#include <memory>
#include <string>
#include <grpc++/grpc++.h>

#include "net/master_env.h"
#include "net/worker_env.h"
#include "net/grpc_worker_cache.h"
#include "net/grpc_channel.h"
#include "net/server_lib.h"
#include "net/grpc_server_lib.h"
#include "net/grpc_master_service.h"
#include "net/async_service_interface.h"
#include "net/grpc_worker_service.h"
#include "net/master_session.h"
#include "net/platform_env.h"

namespace oneflow{

GrpcServer::GrpcServer(ServerDef& server_def) : 
    server_def_(server_def){}

GrpcServer::~GrpcServer(){
  delete master_service_;
  delete worker_service_;
}

void GrpcServer::Init(){
  std::cout<<"Init start------"<<std::endl;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:50051", ::grpc::InsecureServerCredentials());
  //master service and worker service use the same builder
  master_service_ = NewGrpcMasterService(&builder);
  worker_service_ = NewGrpcWorkerService(&builder);
  server_ = builder.BuildAndStart();
  //master servie and woker service use the same channle_cache
  GrpcChannelSpec channel_spec;
  for(auto& job : server_def_.cluster().job()) {
    int max_task_id = -1;
    for(auto& task : job.tasks()){
      max_task_id = std::max(max_task_id, task.first);
    }
    std::vector<std::string> host_ports(max_task_id + 1);
    for(auto& task : job.tasks()) {
      if(job.name() == server_def_.job_name() &&
         task.first == server_def_.task_index()) {
        host_ports[task.first] = task.second;
      }
      else {
        host_ports[task.first] = task.second;
      }
    }
    channel_spec.AddHostPortsJob(job.name(), host_ports, host_ports.size());
  }
  std::unique_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()));
  worker_env_.worker_cache = NewGrpcWorkerCache(channel_cache.release());
  master_env_.worker_cache = worker_env_.worker_cache;
  //master service use MasterSession;
  master_env_.master_session_factory = NewMasterSession;
  std::cout<<"Init end------"<<std::endl;
}

void GrpcServer::Start(){
  master_thread_.reset(
      StartThread(ThreadOptions(), "master_service", [this] {master_service_->HandleRPCsLoop();}));
  std::cout<<"start master service------"<<std::endl;
  worker_thread_.reset(
      StartThread(ThreadOptions(), "worker_service", [this] {worker_service_->HandleRPCsLoop();})); 
  std::cout<<"start worker service------"<<std::endl;
}

void GrpcServer::Stop() {
  server_->Shutdown();
  master_service_->Shutdown();
  worker_service_->Shutdown();
}

void GrpcServer::Join() {}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  return NewHostPortGrpcChannel;
}

void GrpcServer::Create(ServerDef& server_def,std::unique_ptr<ServerInterface>* out_server) {
  std::cout<<"create GrpcServer"<<std::endl;
  std::unique_ptr<GrpcServer> ret(new GrpcServer(server_def));
  ret->Init();
  *out_server = std::move(ret);
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  void NewServer(ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) {
    GrpcServer::Create(server_def, out_server);
  }
};

}//end namespace

}//end namespace oneflow
