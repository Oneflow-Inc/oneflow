/*
 * grpc_server_service.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_SERVER_SERVICE_H
#define GRPC_SERVER_SERVICE_H

#include <grpc++/grpc++.h>


#include "distributed_runtime/worker_service.pb.h"
#include "distributed_runtime/grpc_call.h"
#include "distributed_runtime/grpc_worker_service_impl.h"
#include "distributed_runtime/worker.h"

namespace oneflow {
 
using ::grpc::ServerBuilder;

class GrpcWorkerService {
  public:
    GrpcWorkerService(::grpc::ServerBuilder* builder) {
      builder->RegisterService(&worker_service_);
      cq_ = builder->AddCompletionQueue();
    }

    ~GrpcWorkerService() {}

#define ENQUEUE_REQUEST(method)                                           \
  do {                                                                    \
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,          \
           method##Request, method##Response>::                           \
      EnqueueRequestForMethod(                                            \
          &worker_service_, cq_.get(),                                    \
          static_cast<int>(GrpcWorkerMethod::k##method),            \
          &GrpcWorkerService::method##Handler);                           \
  } while (0)


    void HandleRPCsLoop() {
      ENQUEUE_REQUEST(GetMachineDesc);
      ENQUEUE_REQUEST(GetMemoryDesc);
      void* tag;
      bool ok;
      while(cq_->Next(&tag, &ok)) {
        UntypedCall<GrpcWorkerService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
        if(callback_tag) callback_tag->OnCompleted(this);
      }//while
    }

    std::unique_ptr<::grpc::ServerCompletionQueue> cq_;

    grpc::WorkerService::AsyncService worker_service_;

    Worker* wi_;
    ::grpc::Status status;

  private:
    template <class RequestMessage, class ResponseMessage>
    using WorkerCall = Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
                             RequestMessage, ResponseMessage>;

    void GetMachineDescHandler(WorkerCall<GetMachineDescRequest,
                                                GetMachineDescResponse>* call) {
      //TODO[xiaoshu]
      wi_->GetMachineDesc(&call->request, &call->response);
      call->SendResponse(status);
    }

    void GetMemoryDescHandler(WorkerCall<GetMemoryDescRequest, 
                                         GetMemoryDescResponse>* call) {
      //TODO[xiaoshu]
      wi_->GetMemoryDesc(&call->request, &call->response);
      call->SendResponse(status);
    }
    
    void SendMessageHandler(WorkerCall<SendMessageRequest,
                                       SendMessageResponse>* call) {
      //TODO[xiaoshu]
      //call function in networker/grpc/worker.h 
    }

    void ReadDataHandler(WorkerCall<ReadDataRequest, 
                                    ReadDataResponse>* call) {
      //TODO
      //call function in networker/grpc/worker.h
    }
};

}

#endif /* !GRPC_SERVER_SERVICE_H */
