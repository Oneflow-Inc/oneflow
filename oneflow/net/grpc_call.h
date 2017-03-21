#ifndef GRPC_CALL_H_
#define GRPC_CALL_H_

#include "common/refcount.h"
#include "grpc++/grpc++.h"
#include "grpc++/server_builder.h"

namespace oneflow {

template <class Service>
class UntypedCall : public core::RefCounted {
 public:
  virtual ~UntypedCall() {}

 class Tag {
  public:
   
 };
};

template <class Service, class GrpcService, class RequestMessage, class ResponseMessage>
class Call : public UntypedCall<Service> {
 public:

  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*); 

  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function), responder_(&ctx_) {}

  static void EnqueueRequest(GrpcService* grpc_service,
                            ::grpc::ServerCompletionQueue* cq,
		            EnqueueFunction enqueue_function,
			    HandleRequestFunction handle_request_function,
			    bool supports_cancel){
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(handle_request_function);
  }
  
  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;
};


}

#endif
