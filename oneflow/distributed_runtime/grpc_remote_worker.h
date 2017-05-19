#ifndef GRPC_REMOTE_WORKER_H
#define GRPC_REMOTE_WORKER_H

#include "grpc++/grpc++.h"
#include "distributed_runtime/worker.h"

#include "distributed_runtime/grpc_worker_service_impl.h"
#include "distributed_runtime/worker_service.pb.h"

namespace oneflow {

typedef std::function<void()> Callback;

class GrpcRemoteWorker {
  public:
    GrpcRemoteWorker(std::shared_ptr<::grpc::Channel> channel,
                     ::grpc::CompletionQueue* completion_queue)
    : channel_(channel),
      cq_(completion_queue),
      getmachinedesc_(Method(GrpcWorkerMethod::kGetMachineDesc)),
      getmemorydesc_(Method(GrpcWorkerMethod::kGetMemoryDesc)),
      sendmessage_(Method(GrpcWorkerMethod::kSendMessage)), 
      readdata_(Method(GrpcWorkerMethod::kReadData)){}


    ~GrpcRemoteWorker() {}

    void GetMachineDescAsync(GetMachineDescRequest* request,
                             GetMachineDescResponse* response) {
      IssueRequest(request, response, getmachinedesc_);
    }

    void GetMemoryDescAsync(GetMemoryDescRequest* request,
                            GetMemoryDescResponse* response) {
      IssueRequest(request, response, getmemorydesc_);
    }

    void SendMessageAsync(SendMessageRequest* request,
                          SendMessageResponse* response) {
      IssueRequest(request, response, sendmessage_);
    }

    void ReadDataAsync(ReadDataRequest* request,
                       ReadDataResponse* response) {
      IssueRequest(request, response, readdata_);
    }

  private:
    template <class RequestMessage, class ResponseMessage>
    class RPCState {
      public:
        RPCState(::grpc::ChannelInterface* channel, ::grpc::CompletionQueue* cq, 
                 const ::grpc::RpcMethod& method, const RequestMessage& request)
          : reader_(channel, cq, method, context_, request) {}

        ~RPCState() {}

        void StartRPC(ResponseMessage* response) {
          reader_.Finish(response, &status_, this);
        }

        void OnCompleted(bool ok) {
          if(ok) done_(status_);//will be called in grpc_worker_cache.h
        }

      private:
        ::grpc::ClientContext* context_;
        ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
        ::grpc::Status status_;
        Callback done_;

    };

    template <class RequestMessage, class ResponseMessage>
    void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                      const ::grpc::RpcMethod& method) {
      auto state = new RPCState<RequestMessage, ResponseMessage>(
          channel_.get(), cq_, method, *request);
      state->StartRPC(response);
    }//IssueRequest

    ::grpc::RpcMethod Method(GrpcWorkerMethod id) {
      return ::grpc::RpcMethod(GrpcWorkerMethodName(id), ::grpc::RpcMethod::NORMAL_RPC, channel_);
    }

    std::shared_ptr<::grpc::Channel> channel_;
    ::grpc::CompletionQueue* cq_;

    const ::grpc::RpcMethod getmachinedesc_;
    const ::grpc::RpcMethod getmemorydesc_;
    const ::grpc::RpcMethod sendmessage_;
    const ::grpc::RpcMethod readdata_;
};


}

#endif /* !GRPC_REMOTE_WORKER_H */
