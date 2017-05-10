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

void GrpcServer::InitTopology(oneflow::Topology topology, std::string& FilePath) {
  {
    std::ifstream confFile(FilePath);
    google::protobuf::io::IstreamInputStream in(&confFile);
    if(!google::protobuf::TextFormat::Parse(&in, &topology)) {
      confFile.close();
    }
    for(auto& pair : topology.pair()){
      vec_.push_back(pair.dst());
    }
  }
}

void GrpcServer::StartService() {

}

void GrpcServer::InitContext() {

}

}


