/*
 * grpc_init_service_impl.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_worker_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"

namespace oneflow {

const char* GrpcWorkerMethodName(GrpcWorkerMethod id) {
  switch(id) {
    case GrpcWorkerMethod::kGetMachineDesc:
      return "/oneflow.WorkerService/GetMachineDesc";
    case GrpcWorkerMethod::kGetMemoryDesc:
      return "/oneflow.WorkerService/GetMemoryDesc";
  }
}

namespace grpc {

WorkerService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
                                           ::grpc::RpcMethod::NORMAL_RPC,
                                           nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}

}//namespace grpc

}//namespace oneflow



