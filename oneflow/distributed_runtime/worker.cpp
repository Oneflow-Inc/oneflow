/*
 * worker.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/worker.h"

#include <iostream>
#include <fstream>

#include "distributed_runtime/worker.pb.h"

namespace oneflow {

Worker::Worker() {}

void Worker::GetMachineDesc(GetMachineDescRequest* request,
                       GetMachineDescResponse* response) {
   machine_desc_.machine_id = request->machine_desc().machine_id();
   machine_desc_.ip = request->machine_desc().ip();
   machine_desc_.port = request->machine_desc().port();

   oneflow::MachineDesc machine_desc_for_response;
   machine_desc_file_path = "./machine_desc.txt";
   ParseToProto(machine_desc_for_response, machine_desc_file_path);
   response->mutable_machine_desc()->set_machine_id(machine_desc_for_response.machine_id());
   response->mutable_machine_desc()->set_ip(machine_desc_for_response.ip());
   response->mutable_machine_desc()->set_port(machine_desc_for_response.port());
}

void Worker::GetMemoryDesc(GetMemoryDescRequest* request,
                           GetMemoryDescResponse* response) {
  memory_desc_.machine_id = request->memory_desc().machine_id();
  memory_desc_.memory_address = request->memory_desc().memory_address();
  memory_desc_.remote_token = request->memory_desc().remoted_token();

  oneflow::MemoryDesc memory_desc_for_resp;
  memory_desc_file_path = "./memory_desc.txt";
  ParseToProto(memory_desc_for_resp, memory_desc_file_path);
  response->mutable_memory_desc()->set_machine_id(memory_desc_for_resp.machine_id());
  response->mutable_memory_desc()->set_memory_address(memory_desc_for_resp.memory_address());
  response->mutable_memory_desc()->set_remoted_token(memory_desc_for_resp.remoted_token());
}

void Worker::SendMessage(SendMessageRequest* request,
                 SendMessageResponse* response) {
  //TODO
  //get message from request 
}

void Worker::ReadData(ReadDataRequest* request,
              ReadDataResponse* response) {
  //TODO
  //fill data in respone
  
}

template <typename ProtoMessage>
void Worker::ParseToProto(ProtoMessage& proto_type, std::string& file_name) {
  std::ifstream input_file(file_name); 
  google::protobuf::io::IstreamInputStream proto_file(&input_file);
  if(!google::protobuf::TextFormat::Parse(&proto_file, &proto_type)) {
    input_file.close();
  }//end if
  
}//end Parsetoproto

}



