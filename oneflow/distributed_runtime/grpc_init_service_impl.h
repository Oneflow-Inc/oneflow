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

    class Stub GRPC_FINAL : public StubInterface {
      public:
        Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
        ::grpc::Status exchange_machine_id(::grpc::ClientContext* context,
                                          const ::oneflow::Node& request,
                                          ::oneflow::Node* response) GRPC_OVERRIDE;
        ::grpc::Status exchange_memory_desc(::grpc::ClientContext* context,
                                          const ::oneflow::mem_desc& request,
                                          ::oneflow::mem_desc* response) GRPC_OVERRIDE;

      private:
        std::shared_ptr<::grpc::ChannelInterface> channel_;
        const ::grpc::RpcMethod rpcmethod_exchange_machine_id_;
        const ::grpc::RpcMethod rpcmethod_exchange_memory_desc_;
    }; 

    static std::unique_ptr<Stub> NewStub(
        const std::shared_ptr< ::grpc::ChannelInterface>& channel,
        const ::grpc::StubOptions& options = ::grpc::StubOptions());
  
    class Service : public ::grpc::Service {
      public:
        Service();
        virtual ~Service();

        void Requestexchange_machine_id(::grpc::ServerContext* context, ::oneflow::    Node* request,
            ::grpc::ServerAsyncResponseWriter< ::oneflow::Node>* response,
            ::grpc::CompletionQueue* new_call_cq,
            ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
          ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
        }

        void Requestexchange_memory_desc(::grpc::ServerContext* context, ::oneflow::mem_desc* request,
            ::grpc::ServerAsyncResponseWriter< ::oneflow::mem_desc>* response, ::grpc::CompletionQueue* new_call_cq,
            ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
          ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
        }
    };
};

}

}

  

#endif /* !GRPC_INIT_SERVICE_IMPL_H */
