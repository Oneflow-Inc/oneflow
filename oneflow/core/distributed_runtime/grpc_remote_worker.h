#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_WORKER_H_

#include "grpc++/grpc++.h"
#include "oneflow/core/distributed_runtime/worker.h"

//#include "oneflow/core/distributed_runtime/tensor_coding.h"

#include "oneflow/core/distributed_runtime/grpc_worker_service_impl.h"
#include "oneflow/core/distributed_runtime/worker_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_client_cq_tag.h"

namespace oneflow {

class GrpcRemoteWorker {
 public:
  GrpcRemoteWorker(std::shared_ptr<::grpc::Channel> channel,
                   ::grpc::CompletionQueue* completion_queue)
    : channel_(std::move(channel)),
      stub_(grpc::WorkerService::NewStub(channel)),
      cq_(completion_queue),
      sendmessage_(Method(GrpcWorkerMethod::kSendMessage)),
      readdata_(Method(GrpcWorkerMethod::kReadData)) {}

    ~GrpcRemoteWorker() {}

    ::tensorflow::Status GetStatus(const GetStatusRequest* request,
                                   GetStatusResponse* response) {
      ::grpc::ClientContext ctx;
      return FromGrpcStatus(stub_->GetStatus(&ctx, *request, response));
    }

    ::tensorflow::Status GetMachineDesc(const GetMachineDescRequest* request,
                        GetMachineDescResponse* response) {
      ::grpc::ClientContext ctx;
      return FromGrpcStatus(stub_->GetMachineDesc(&ctx, *request, response));
    }

    ::tensorflow::Status GetMemoryDesc(const GetMemoryDescRequest* request,
                       GetMemoryDescResponse* response) {
      ::grpc::ClientContext ctx;
      return FromGrpcStatus(stub_->GetMemoryDesc(&ctx, *request, response)); 
    }

    ::tensorflow::Status SendTaskGraph(const SendTaskGraphRequest* request,
                       SendTaskGraphResponse* response) {
      ::grpc::ClientContext ctx;
      return FromGrpcStatus(stub_->SendTaskGraph(&ctx, *request, response));
    }

    void SendMessageAsync(const SendMessageRequest* request,
                          SendMessageResponse* response,
                          StatusCallback done) {
      IssueRequest(request, response, sendmessage_, std::move(done));
    }

    void ReadDataAsync(ReadDataRequest* request,
                       ReadDataResponse* response,
                       StatusCallback done) {
      // callback done is passed in by NetActor
      IssueRequest(request, response, readdata_, std::move(done));
    }

 public:
  std::unique_ptr<grpc::WorkerService::Stub> stub_;

  template <class RequestMessage, class ResponseMessage>
  class RPCState final : public GrpcClientCQTag {
   public:
    RPCState(::grpc::ChannelInterface* channel, ::grpc::CompletionQueue* cq,
             const ::grpc::RpcMethod& method, const RequestMessage& request,
             StatusCallback done)
      : reader_(channel, cq, method, InitContext(), request),
        done_(std::move(done)) {}

    ~RPCState() {}

    void StartRPC(ResponseMessage* response) {
      reader_.Finish(response, &status_, this);
    }

    void OnCompleted(bool ok) override {
      if (ok) {
        done_(FromGrpcStatus(status_));
        delete this;
      }
    }

   private:
    ::grpc::ClientContext context_;
    ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
    ::grpc::Status status_;
    StatusCallback done_;

    ::grpc::ClientContext* InitContext() {
      return &context_;
    }
  };  // class RPCState

  template <class RequestMessage, class ResponseMessage>
  void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                    const ::grpc::RpcMethod& method, StatusCallback done) {
    auto state = new RPCState<RequestMessage, ResponseMessage>(
      channel_.get(), cq_, method, *request, std::move(done));
    state->StartRPC(response);
  }  // IssueRequest

  ::grpc::RpcMethod Method(GrpcWorkerMethod id) {
    return ::grpc::RpcMethod(GrpcWorkerMethodName(id), 
                             ::grpc::RpcMethod::NORMAL_RPC, 
                             channel_);
  }

  std::shared_ptr<::grpc::Channel> channel_;
  ::grpc::CompletionQueue* cq_;

  const ::grpc::RpcMethod sendmessage_;
  const ::grpc::RpcMethod readdata_;
};  // class GrpcRemoteWorker

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_WORKER_H_
