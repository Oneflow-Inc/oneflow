#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CALL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CALL_H_

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/default/mutex.h"

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/call.h"
#include "grpc++/support/byte_buffer.h"



namespace oneflow {

template <class Service>
class UntypedCall : public tensorflow::core::RefCounted {
 public:
  virtual ~UntypedCall() {}

  virtual void RequestReceived(Service* service, bool ok) = 0;

  virtual void RequestCancelled(Service* service, bool ok) = 0;

  class Tag {
   public:
    enum Callback {kRequestReceived, kResponseSent, kCancelled};

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

    void OnCompleted(Service* service, bool ok) {
      switch (callback_) {
        case kRequestReceived:
          call_->RequestReceived(service, ok);
          break;
        case kResponseSent:
          break;
        case kCancelled:
          call_->RequestCancelled(service, ok);
          break;
      }
        call_->Unref();
      }

   private:
    UntypedCall* const call_;
    Callback callback_;
  };  // Tag
};  // Untypedcall

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

  void RequestReceived(Service* service, bool ok) override {
    if (ok) {
      this->Ref();
      (service->*handle_request_function_)(this);
    }
  }

  void SendResponse(::grpc::Status status) {
    this->Ref();
    responder_.Finish(response, status, &response_sent_tag_);
    this->Unref();
  }

  void RequestCancelled(Service* service, bool ok) override {
    if (ctx_.IsCancelled()) {
      ::tensorflow::mutex_lock l(mu_);
      if (cancel_callback_) {
        cancel_callback_();
      }
    }
  }

  void SetCancelCallback(std::function<void()> callback) {
    ::tensorflow::mutex_lock l(mu_);
    cancel_callback_ = std::move(callback);
  }

  void ClearCancelCallback() {
    tensorflow::mutex_lock l(mu_);
    cancel_callback_ = nullptr;
  }

  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function,
                             bool supports_cancel) {
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
          handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }
    (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
  }  // EnqueueRequest

  static void EnqueueRequestForMethod(GrpcService* grpc_service,
                ::grpc::ServerCompletionQueue* cq,
                int method_id,
                HandleRequestFunction handle_request_function,
                bool supports_cancel) {
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }
    grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                    &call->responder_, cq, cq,
                                    &call->request_received_tag_);
  }  // EnqueueRequestForMethod

  RequestMessage request;
  ResponseMessage response;

 private:
  void RegisterCancellationHandler() {
    this->Ref();
    ctx_.AsyncNotifyWhenDone(&cancelled_tag_);
  }

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;

  typedef typename UntypedCall<Service>::Tag Tag;
  Tag request_received_tag_{this, Tag::kRequestReceived};
  Tag response_sent_tag_{this, Tag::kResponseSent};
  Tag cancelled_tag_{this, Tag::kCancelled};

  ::tensorflow::mutex mu_;
  std::function<void()> cancel_callback_ GUARDED_BY(mu_);
};  // class Call

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CALL_H_ 
