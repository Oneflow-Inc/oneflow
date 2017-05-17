/*
 * grpc_call.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_CALL_H
#define GRPC_CALL_H

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/server_builder.h"

namespace oneflow {
 
template <class Service>
class UntypedCall {
  public:
    virtual ~UntypedCall() {}

    virtual void RequestReceived(Service* service) = 0;

    class Tag {
      public:
        enum Callback {kRequestReceived, kReponseSent};
        Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

        void OnCompleted(Service* service) {
          switch(callback_) {
            case kRequestReceived:
              call_->RequestReceived(service);
          }
        }
      private:
        UntypedCall* const call_;
        Callback callback_;
    };
}; 

template <class Service, class GrpcService, 
          class RequestMessage, class ResponseMessage>
class Call : public UntypedCall<Service> {
  public:
    using EnqueueFunction = void (GrpcService::*)(
        ::grpc::ServerContext*, RequestMessage*,
        ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
        ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

    using HandleRequestFunction = void (Service::*)(
        Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

    Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function),
        responder_(&ctx_) {}
    virtual ~Call() {}

    void RequestReceived(Service* service) override {
      (service->*handle_request_function_)(this);
    }
    void SendResponse(::grpc::Status status) {
      responder_.Finish(response, status, &response_sent_tag_);
    }

    static void EnqueueRequest(GrpcService* grpc_service,
                               ::grpc::ServerCompletionQueue* cq,
                               EnqueueFunction enqueue_function,
                               HandleRequestFunction handle_request_function) {
      auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage> (handle_request_function);
      (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                        &call->responder_, cq, cq,
                                        &call->request_received_tag_); 
    }

    static void EnqueueRequestForMethod(GrpcService* grpc_service,
                                        ::grpc::ServerCompletionQueue* cq,
                                        int method_id, 
                                        HandleRequestFunction handle_request_function) {
      auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(handle_request_function);
      grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
    }

    RequestMessage request;
    ResponseMessage response;

  private:
    HandleRequestFunction handle_request_function_;
    ::grpc::ServerContext ctx_;
    ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;
  
    typedef typename UntypedCall<Service>::Tag Tag;
    Tag request_received_tag_{this, Tag::kRequestReceived};
    Tag response_sent_tag_{this, Tag::kResponseSent};

};
 
}


#endif /* !GRPC_CALL_H */
