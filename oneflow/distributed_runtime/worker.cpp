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
  response->set_tmp(1);
}

void Worker::GetMemoryDesc(GetMemoryDescRequest* request,
                           GetMemoryDescResponse* response) {
  response->set_tmp(2);
}

void SendMessage(SendMessageRequest* request,
                 SendMessageResponse* response) {
  //TODO
  //call function in network/grpc/grpc_worker.cc
}

void ReadData(ReadDataRequest* request,
              ReadDataResponse* response) {
  //TODO
  //call function in network/grpc/grpc_worker.cc
}

}



