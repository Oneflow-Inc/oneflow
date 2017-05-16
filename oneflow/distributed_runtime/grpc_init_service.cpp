/*
 * grpc_server_service.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_init_service.h"

namespace oneflow {

GrpcInitService::GrpcInitService(::grpc::ServerBuilder* builder) {
  builder->RegisterService(&service_);
  cq_ = builder->AddCompletionQueue();
}

GrpcInitService::~GrpcInitService() {}

void GrpcInitService::HandleRPCsLoop() {

}



}



