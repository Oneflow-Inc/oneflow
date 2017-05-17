/*
 * grpc_remote_worker.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_REMOTE_WORKER_H
#define GRPC_REMOTE_WORKER_H

#include "grpc++/grpc++.h"
#include "distributed_runtime/worker_service.pb.h"


namespace oneflow {

class GrpcRemoteWorker {
  public:
    GrpcRemoteWorker(::grpc::Channel channel,
                     ::grpc::CompletionQueue* completion_queue)
    : channel_(channel),
      cq_(completion_queue) {}
    ~GrpcRemoteWorker() {}

    void GetMachineDescAsync(GetMachineDescRequest* request,
                             GetMachineDescResponse* response) {
      IssueRequest(request, response, getmachinedesc_);
    }

    void GetMemoryDescAsync(GetMemoryDescRequest* request,
                            GetMemoryDescResponse* response) {
      IssueRequest(request, response, getmemorydesc_);
    }


  private:

    template <class RequestMessage, class ResponseMessage>
    class RPCState final {
      public:
        RPCState(::grpc::CompletionQueue* cq, const ::grpc::RpcMethod& method, const RequestMessage& request)
          : reader_(channel, cq, method, context_, request) {}
        ~RPCState() {}

        void StartRPC(ResponseMessage* response) {
          reader_.Finish(response, &status_, this);
        }

      private:
        ::grpc::ClientContext context_;
        ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
        ::grpc::Status status_;
    };

    template <class RequestMessage, class ResponseMessage>
    void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                      const ::grpc::RpcMethod& method) {
      auto state = new RPCState<RequestMessage, ResponseMessage>(
          cq_, method, *request);
      state->StartRPC(response);
    
    }

    ::grpc::Channel channel_;

    const ::grpc::RpcMethod getmachinedesc_;
    const ::grpc::RpcMethod getmemorydesc_;
};


}

#endif /* !GRPC_REMOTE_WORKER_H */
