#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_WORKER_H_

#include "grpc++/grpc++.h"
#include "oneflow/core/distributed_runtime/worker.h"

//#include "oneflow/core/distributed_runtime/tensor_coding.h"

#include "oneflow/core/distributed_runtime/grpc_worker_service_impl.h"
#include "oneflow/core/distributed_runtime/worker_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_client_cq_tag.h"

namespace oneflow {

typedef std::function<void()> Callback;

class GrpcRemoteWorker {
 public:
  GrpcRemoteWorker(std::shared_ptr<::grpc::Channel> channel,
                   ::grpc::CompletionQueue* completion_queue)
    : channel_(channel),
      stub_(grpc::WorkerService::NewStub(channel)),
      cq_(completion_queue),
      getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
      getmachinedesc_(Method(GrpcWorkerMethod::kGetMachineDesc)),
      getmemorydesc_(Method(GrpcWorkerMethod::kGetMemoryDesc)),
      sendtaskgraph_(Method(GrpcWorkerMethod::kSendTaskGraph)),
      sendmessage_(Method(GrpcWorkerMethod::kSendMessage)),
      readdata_(Method(GrpcWorkerMethod::kReadData)) {}

    ~GrpcRemoteWorker() {}

    void GetMachineDesc(GetMachineDescRequest request,
                        GetMachineDescResponse response) {
      ::grpc::ClientContext ctx;
      stub_->GetMachineDesc(&ctx, request, &response);
    }

    void GetMemoryDesc(GetMemoryDescRequest request,
                       GetMemoryDescResponse response) {
      ::grpc::ClientContext ctx;
      stub_->GetMemoryDesc(&ctx, request, &response); 
    }

    void SendTaskGraph(SendTaskGraphRequest request,
                       SendTaskGraphResponse response) {
      ::grpc::ClientContext ctx;
      stub_->SendTaskGraph(&ctx, request, &response);
    }

    void SendMessageAsync(SendMessageRequest* request,
                          SendMessageResponse* response,
                          Callback done) {
      IssueRequest(request, response, sendmessage_, std::move(done));
    }

    void ReadDataAsync(ReadDataRequest* request,
                       TensorResponse* response,
                       Callback done) {
      ReadDataRequest* req_copy = nullptr;
      req_copy = new ReadDataRequest;
      *req_copy = *request;

      Callback wrapper_done;
      const Callback* cb_to_use;

      wrapper_done = [this, request, req_copy, response, done]() {
        //int64_t step_id = request->step_id();
        done();
      };
      cb_to_use = &wrapper_done;
      // callback done is passed in by NetActor
      IssueRequest(request, response, readdata_, std::move(done));
    }

 private:
  std::unique_ptr<grpc::WorkerService::Stub> stub_;

  template <class RequestMessage, class ResponseMessage>
  class RPCState : GrpcClientCQTag {
   public:
    RPCState(::grpc::ChannelInterface* channel, ::grpc::CompletionQueue* cq, 
             const ::grpc::RpcMethod& method, const RequestMessage& request,
             Callback done)
      : reader_(channel, cq, method, context_, request),
        done_(done) {}

    ~RPCState() {}

    void StartRPC(ResponseMessage* response) {
      reader_.Finish(response, &status_, this);
    }

    void OnCompleted(bool ok) {
      if(ok) done_();
    }

    private:
     ::grpc::ClientContext* context_;
     ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
     Callback done_;
  };  // class RPCState

  template <class RequestMessage, class ResponseMessage>
  void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                    const ::grpc::RpcMethod& method, Callback done) {
    auto state = new RPCState<RequestMessage, ResponseMessage>(
                   channel_.get(), cq_, method, *request, std::move(done));
    state->StartRPC(response);
  }//IssueRequest

  ::grpc::RpcMethod Method(GrpcWorkerMethod id) {
    return ::grpc::RpcMethod(GrpcWorkerMethodName(id), ::grpc::RpcMethod::NORMAL_RPC, channel_);
  }

  std::shared_ptr<::grpc::Channel> channel_;
  ::grpc::CompletionQueue* cq_;

  const ::grpc::RpcMethod getstatus_;
  const ::grpc::RpcMethod getmachinedesc_;
  const ::grpc::RpcMethod getmemorydesc_;
  const ::grpc::RpcMethod sendtaskgraph_;
  const ::grpc::RpcMethod sendmessage_;
  const ::grpc::RpcMethod readdata_;
};//class GrpcRemoteWorker


}//namespace oneflow

#endif /* !GRPC_REMOTE_WORKER_H */
