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
  //TODO

   machine_desc_.machine_id = request->machine_desc().machine_id();
   machine_desc_.ip = request->machine_desc().ip();
   machine_desc_.port = request->machine_desc().port();
   oneflow::MachineDesc machine_desc_for_response;
   machine_desc_file_path = "./machie_desc.txt";
   ParseToProto(machine_desc_for_response, machine_desc_file_path);
  //get message from request.
}

void Worker::GetMemoryDesc(GetMemoryDescRequest* request,
                           GetMemoryDescResponse* response) {
  //TODO
  memory_desc_.machine_id = request->memory_desc().machine_id();
  memory_desc_.memory_address = request->memory_desc().memory_address();
  memory_desc_.remote_token = request->memory_desc().remoted_token();
  //get message from request
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



