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

#include "distributed_runtime/oneflow_init.pb.h"

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}//namespace grpc

namespace oneflow {
namespace grpc{

class InitService GRPC_FINAL {
  public:
    class StubInterface {
      public:
        virtual ~StubInterface() {}
        virtual ::grpc::Status exchange_machine_id(::grpc::ClientContext* context,
                                                   const ::oneflow::Node& request,
                                                   ::oneflow::Node* response) = 0;
        virtual ::grpc::Status exchange_memory_desc(::grpc::ClientContext* context,
                                                    const ::oneflow::mem_desc& request,
                                                    ::oneflow::mem_desc* response) = 0;

    };

    

};

}

}

  

#endif /* !GRPC_INIT_SERVICE_IMPL_H */
