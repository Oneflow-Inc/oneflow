/*
 * grpc_init_service_impl.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_INIT_SERVICE_IMPL_H
#define GRPC_INIT_SERVICE_IMPL_H

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

#include "distributed_runtime/worker_service.pb.h"

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}//namespace grpc

namespace oneflow {

enum class GrpcWorkerMethod {
  kGetMachineDesc,
  kGetMemoryDesc,
};
static const int kGrpcNumWorkerMethods = 
  static_cast<int>(GrpcWorkerMethod::kGetMemoryDesc) + 1;

const char* GrpcWorkerMethodName(GrpcWorkerMethod id);

namespace grpc{

class WorkerService GRPC_FINAL {
  public:
    class AsyncService : public ::grpc::Service {
      public:
        AsyncService();
        virtual ~AsyncService();
        
        using ::grpc::Service::RequestAsyncUnary;
    };
};

}//namespace grpc
}//namespace oneflow

  

#endif /* !GRPC_INIT_SERVICE_IMPL_H */
