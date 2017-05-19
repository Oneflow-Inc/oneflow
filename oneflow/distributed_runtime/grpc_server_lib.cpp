/*
 * grpc_server_init.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_server_lib.h"

#include <iostream>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "distributed_runtime/grpc_remote_worker.h"
#include "distributed_runtime/grpc_worker_session.h"


namespace oneflow {

using ::grpc::ServerBuilder;

GrpcServer::GrpcServer() {}
GrpcServer::~GrpcServer() {}

void GrpcServer::InitTopology(oneflow::Topology topology, std::string& TopologyFilePath, oneflow::MachineList machine, std::string& MachineListFilePath ) {
  std::ifstream topologyFile(TopologyFilePath);
  google::protobuf::io::IstreamInputStream inTopologyFile(&topologyFile);
  if(!google::protobuf::TextFormat::Parse(&inTopologyFile, &topology)) {
    topologyFile.close();
  }

  for(auto& pair : topology.pair()){
    vec_.push_back(pair.dst());
    pair_map_.insert({pair.src(), pair.dst()});
  }
  
  std::ifstream resourceFile(MachineListFilePath);
  google::protobuf::io::IstreamInputStream inResourceFile(&resourceFile);
  if(!google::protobuf::TextFormat::Parse(&inResourceFile, &machine)) {
    resourceFile.close();
  }

  for(auto& m : machine.machine_list()) {
    machine_desc md;
    md.id = m.id();
    md.name = m.name();
    md.ip = m.ip();
    md.port = m.port(); 
    machine_list_.insert({md.id, md});
  }

  CreateChannelCache(); 

}

void GrpcServer::CreateChannelCache() {
  for(auto& pair : pair_map_){
    std::string src = pair.first;
    std::string dst = pair.second;
    machine_desc src_mdesc = machine_list_[src];
    std::string src_ip = src_mdesc.ip;
    std::string src_port = src_mdesc.port;
    std::string src_ip_port = src_ip + ":" + src_port;

    machine_desc dst_mdesc = machine_list_[dst];
    std::string dst_ip = dst_mdesc.ip;
    std::string dst_port = dst_mdesc.port;
    std::string dst_ip_port = dst_ip + ":" + dst_port;

    std::shared_ptr<::grpc::Channel> to_dst_channel = 
      ::grpc::CreateChannel(dst_ip_port, ::grpc::InsecureChannelCredentials());
    channel_map_.insert({src, to_dst_channel}); 

    std::shared_ptr<::grpc::Channel> to_src_channel = 
      ::grpc::CreateChannel(src_ip_port, ::grpc::InsecureChannelCredentials());
    channel_map_.insert({dst, to_src_channel});
  }//end for
}//end CreateChannelCache

void GrpcServer::StartService() {
  std::string server_address("0.0.0.0:50051");
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  worker_service_ = new GrpcWorkerService(&builder);
  server_ = builder.BuildAndStart();
  
  worker_service_->HandleRPCsLoop();
}

void GrpcServer::InitContext() {

}

}


