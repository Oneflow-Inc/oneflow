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



