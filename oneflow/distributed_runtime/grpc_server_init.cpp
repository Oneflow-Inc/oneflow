/*
 * grpc_server_init.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_server_init.h"

#include <iostream>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>


namespace oneflow {

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
   
  }
}

void GrpcServer::StartService() {

}

void GrpcServer::InitContext() {

}

}


