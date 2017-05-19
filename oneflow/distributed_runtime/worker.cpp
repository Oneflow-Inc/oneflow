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



