/*
 * worker.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/worker.h"
#include "distributed_runtime/worker.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace oneflow {

Worker::Worker() {}

void Worker::GetMachineDesc(GetMachineDescRequest* request,
                       GetMachineDescResponse* response) {
  //TODO

   machine_desc_.machine_id = request->machine_desc().machine_id();
   machine_desc_.ip = request->machine_desc().ip();
   machine_desc_.port = request->machine_desc().port();

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

void SendMessage(SendMessageRequest* request,
                 SendMessageResponse* response) {
  //TODO
  //get message from request 
}

void ReadData(ReadDataRequest* request,
              ReadDataResponse* response) {
  //TODO
  //fill data in respone
  
}

}



