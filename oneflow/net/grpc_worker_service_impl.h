/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/impl/codegen/rpc_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/status.h"
#include "grpc++/impl/codegen/stub_options.h"
#include "grpc++/impl/codegen/sync_stream.h"
#include "grpc++/support/byte_buffer.h"

#include "net/grpc_serialization_traits.h"
#include "proto/worker.pb.h"

// Contains potentially large GraphDef.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(oneflow::RegisterGraphRequest);
// Contains potentially large TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(oneflow::RunGraphRequest);
// Contains potentially large StepStats, TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(oneflow::RunGraphResponse);

namespace oneflow {
}  // namespace oneflow

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace oneflow {

namespace grpc {

// Implementation of `oneflow.WorkerService`, based on the
// definition in "//oneflow/core/protobuf/worker_service.proto",
// and the gRPC generated stub and service classes.
// See the proto file for the definition of methods and messages.
class WorkerService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::GetStatusResponse>>
    AsyncGetStatus(::grpc::ClientContext* context,
                   const ::oneflow::GetStatusRequest& request,
                   ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::GetStatusResponse>>(
          AsyncGetStatusRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::RegisterGraphResponse>>
    AsyncRegisterGraph(::grpc::ClientContext* context,
                       const ::oneflow::RegisterGraphRequest& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::RegisterGraphResponse>>(
          AsyncRegisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::DeregisterGraphResponse>>
    AsyncDeregisterGraph(::grpc::ClientContext* context,
                         const ::oneflow::DeregisterGraphRequest& request,
                         ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::DeregisterGraphResponse>>(
          AsyncDeregisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::RunGraphResponse>>
    AsyncRunGraph(::grpc::ClientContext* context,
                  const ::oneflow::RunGraphRequest& request,
                  ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::RunGraphResponse>>(
          AsyncRunGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::CleanupGraphResponse>>
    AsyncCleanupGraph(::grpc::ClientContext* context,
                      const ::oneflow::CleanupGraphRequest& request,
                      ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::CleanupGraphResponse>>(
          AsyncCleanupGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::CleanupAllResponse>>
    AsyncCleanupAll(::grpc::ClientContext* context,
                    const ::oneflow::CleanupAllRequest& request,
                    ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::CleanupAllResponse>>(
          AsyncCleanupAllRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::LoggingResponse>>
    AsyncLogging(::grpc::ClientContext* context,
                 const ::oneflow::LoggingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::LoggingResponse>>(
          AsyncLoggingRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::TracingResponse>>
    AsyncTracing(::grpc::ClientContext* context,
                 const ::oneflow::TracingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::oneflow::TracingResponse>>(
          AsyncTracingRaw(context, request, cq));
    }

   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::GetStatusResponse>*
    AsyncGetStatusRaw(::grpc::ClientContext* context,
                      const ::oneflow::GetStatusRequest& request,
                      ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::RegisterGraphResponse>*
    AsyncRegisterGraphRaw(::grpc::ClientContext* context,
                          const ::oneflow::RegisterGraphRequest& request,
                          ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::DeregisterGraphResponse>*
    AsyncDeregisterGraphRaw(::grpc::ClientContext* context,
                            const ::oneflow::DeregisterGraphRequest& request,
                            ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::RunGraphResponse>*
    AsyncRunGraphRaw(::grpc::ClientContext* context,
                     const ::oneflow::RunGraphRequest& request,
                     ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::CleanupGraphResponse>*
    AsyncCleanupGraphRaw(::grpc::ClientContext* context,
                         const ::oneflow::CleanupGraphRequest& request,
                         ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::CleanupAllResponse>*
    AsyncCleanupAllRaw(::grpc::ClientContext* context,
                       const ::oneflow::CleanupAllRequest& request,
                       ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::LoggingResponse>*
    AsyncLoggingRaw(::grpc::ClientContext* context,
                    const ::oneflow::LoggingRequest& request,
                    ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::oneflow::TracingResponse>*
    AsyncTracingRaw(::grpc::ClientContext* context,
                    const ::oneflow::TracingRequest& request,
                    ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::GetStatusResponse>>
    AsyncGetStatus(::grpc::ClientContext* context,
                   const ::oneflow::GetStatusRequest& request,
                   ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::oneflow::GetStatusResponse>>(
          AsyncGetStatusRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::RegisterGraphResponse>>
    AsyncRegisterGraph(::grpc::ClientContext* context,
                       const ::oneflow::RegisterGraphRequest& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<
          ::oneflow::RegisterGraphResponse>>(
          AsyncRegisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReader<
        ::oneflow::DeregisterGraphResponse>>
    AsyncDeregisterGraph(::grpc::ClientContext* context,
                         const ::oneflow::DeregisterGraphRequest& request,
                         ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<
          ::oneflow::DeregisterGraphResponse>>(
          AsyncDeregisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::RunGraphResponse>>
    AsyncRunGraph(::grpc::ClientContext* context,
                  const ::oneflow::RunGraphRequest& request,
                  ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::oneflow::RunGraphResponse>>(
          AsyncRunGraphRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::CleanupGraphResponse>>
    AsyncCleanupGraph(::grpc::ClientContext* context,
                      const ::oneflow::CleanupGraphRequest& request,
                      ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<
          ::oneflow::CleanupGraphResponse>>(
          AsyncCleanupGraphRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::CleanupAllResponse>>
    AsyncCleanupAll(::grpc::ClientContext* context,
                    const ::oneflow::CleanupAllRequest& request,
                    ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::oneflow::CleanupAllResponse>>(
          AsyncCleanupAllRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::LoggingResponse>>
    AsyncLogging(::grpc::ClientContext* context,
                 const ::oneflow::LoggingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::oneflow::LoggingResponse>>(
          AsyncLoggingRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::oneflow::TracingResponse>>
    AsyncTracing(::grpc::ClientContext* context,
                 const ::oneflow::TracingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::oneflow::TracingResponse>>(
          AsyncTracingRaw(context, request, cq));
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader<::oneflow::GetStatusResponse>*
    AsyncGetStatusRaw(::grpc::ClientContext* context,
                      const ::oneflow::GetStatusRequest& request,
                      ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::RegisterGraphResponse>*
    AsyncRegisterGraphRaw(::grpc::ClientContext* context,
                          const ::oneflow::RegisterGraphRequest& request,
                          ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::DeregisterGraphResponse>*
    AsyncDeregisterGraphRaw(::grpc::ClientContext* context,
                            const ::oneflow::DeregisterGraphRequest& request,
                            ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::RunGraphResponse>*
    AsyncRunGraphRaw(::grpc::ClientContext* context,
                     const ::oneflow::RunGraphRequest& request,
                     ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::CleanupGraphResponse>*
    AsyncCleanupGraphRaw(::grpc::ClientContext* context,
                         const ::oneflow::CleanupGraphRequest& request,
                         ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::CleanupAllResponse>*
    AsyncCleanupAllRaw(::grpc::ClientContext* context,
                       const ::oneflow::CleanupAllRequest& request,
                       ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::LoggingResponse>*
    AsyncLoggingRaw(::grpc::ClientContext* context,
                    const ::oneflow::LoggingRequest& request,
                    ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::oneflow::TracingResponse>*
    AsyncTracingRaw(::grpc::ClientContext* context,
                    const ::oneflow::TracingRequest& request,
                    ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    const ::grpc::RpcMethod rpcmethod_GetStatus_;
    const ::grpc::RpcMethod rpcmethod_RegisterGraph_;
    const ::grpc::RpcMethod rpcmethod_DeregisterGraph_;
    const ::grpc::RpcMethod rpcmethod_RunGraph_;
    const ::grpc::RpcMethod rpcmethod_CleanupGraph_;
    const ::grpc::RpcMethod rpcmethod_CleanupAll_;
    const ::grpc::RpcMethod rpcmethod_RecvTensor_;
    const ::grpc::RpcMethod rpcmethod_Logging_;
    const ::grpc::RpcMethod rpcmethod_Tracing_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestGetStatus(
        ::grpc::ServerContext* context, ::oneflow::GetStatusRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::GetStatusResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRegisterGraph(
        ::grpc::ServerContext* context,
        ::oneflow::RegisterGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::RegisterGraphResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestDeregisterGraph(
        ::grpc::ServerContext* context,
        ::oneflow::DeregisterGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<
            ::oneflow::DeregisterGraphResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRunGraph(
        ::grpc::ServerContext* context, ::oneflow::RunGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::RunGraphResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(3, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestCleanupGraph(
        ::grpc::ServerContext* context,
        ::oneflow::CleanupGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::CleanupGraphResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(4, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestCleanupAll(
        ::grpc::ServerContext* context,
        ::oneflow::CleanupAllRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::CleanupAllResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(5, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRecvTensorRaw(
        ::grpc::ServerContext* context,
        ::oneflow::RecvTensorRequest* request,
        ::grpc::ServerAsyncResponseWriter<::grpc::ByteBuffer>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(6, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestLogging(
        ::grpc::ServerContext* context, ::oneflow::LoggingRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::LoggingResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(7, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestTracing(
        ::grpc::ServerContext* context, ::oneflow::TracingRequest* request,
        ::grpc::ServerAsyncResponseWriter<::oneflow::TracingResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(8, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};

}  // namespace grpc

}  // namespace oneflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
