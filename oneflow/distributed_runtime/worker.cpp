/*
 * worker.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/worker.h"

namespace oneflow {

Worker::Worker() {}

void Worker::GetMachineDesc(GetMachineDescRequest* request,
                       GetMachineDescResponse* response) {
  //TODO
  //get message from request.
  response->set_tmp(1);
}

void Worker::GetMemoryDesc(GetMemoryDescRequest* request,
                           GetMemoryDescResponse* response) {
  //TODO
  //get message from request
  response->set_tmp(2);
}

void SendMessage(SendMessageRequest* request,
                 SendMessageResponse* response) {
  //TODO
  //get message from request 
  //and fill response integer 1
}

void ReadData(ReadDataRequest* request,
              ReadDataResponse* response) {
  //TODO
  //fill data in respone
}

}



