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
        virtual ::grpc::Status ExchangeMachineInfo(::grpc::ClientContext* context,
                                                   const ::oneflow::ExchangeMachineInfoRequest& request,
                                                   ::oneflow::ExchangeMachineInfoResponse* response) = 0;
        virtual ::grpc::Status ExchangeMemoryDesc(::grpc::ClientContext* context,
                                                    const ::oneflow::ExchangeMemoryDescRequest& request,
                                                    ::oneflow::ExchangeMemoryDescResponse* response) = 0;
    };

    class Stub GRPC_FINAL : public StubInterface {
      public:
        Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
        ::grpc::Status ExchangeMachineInfo(::grpc::ClientContext* context,
                                          const ::oneflow::ExchangeMachineInfoRequest& request,
                                          ::oneflow::ExchangeMachineInfoResponse* response) GRPC_OVERRIDE;
        ::grpc::Status ExchangeMemoryDesc(::grpc::ClientContext* context,
                                          const ::oneflow::ExchangeMemoryDescRequest& request,
                                          ::oneflow::ExchangeMemoryDescResponse* response) GRPC_OVERRIDE;

      private:
        std::shared_ptr<::grpc::ChannelInterface> channel_;
        const ::grpc::RpcMethod rpcmethod_ExchangeMachineInfo_;
        const ::grpc::RpcMethod rpcmethod_ExchangeMemoryDesc_;
    }; 

    static std::unique_ptr<Stub> NewStub(
        const std::shared_ptr< ::grpc::ChannelInterface>& channel,
        const ::grpc::StubOptions& options = ::grpc::StubOptions());
  
    class Service : public ::grpc::Service {
      public:
        Service();
        virtual ~Service();

        void RequestExchangeMachineInfo(::grpc::ServerContext* context, ::oneflow::ExchangeMachineInfoRequest* request,
            ::grpc::ServerAsyncResponseWriter< ::oneflow::ExchangeMachineInfoResponse>* response,
            ::grpc::CompletionQueue* new_call_cq,
            ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
          ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
        }

        void RequestExchangeMemoryDesc(::grpc::ServerContext* context, ::oneflow::ExchangeMemoryDescRequest* request,
            ::grpc::ServerAsyncResponseWriter< ::oneflow::ExchangeMemoryDescResponse>* response, ::grpc::CompletionQueue* new_call_cq,
            ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
          ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
        }
    };
};

}

}

  

#endif /* !GRPC_INIT_SERVICE_IMPL_H */
