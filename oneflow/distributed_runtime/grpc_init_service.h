/*
 * grpc_server_service.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_SERVER_SERVICE_H
#define GRPC_SERVER_SERVICE_H

#include <grpc++/grpc++.h>
#include "distributed_runtime/oneflow_init.grpc.pb.h"
#include "distributed_runtime/oneflow_init.pb.h"
#include "distributed_runtime/grpc_call.h"
#include "distributed_runtime/grpc_init_service_impl.h"

namespace oneflow {
 
using ::grpc::ServerBuilder;

class GrpcInitService {
  public:
    GrpcInitService(::grpc::ServerBuilder* builder) {
      builder->RegisterService(&service_);
      cq_ = builder->AddCompletionQueue();
    }
    ~GrpcInitService() {}

#define ENQUEUE_REQUEST(method)                                           \
  do {                                                                    \
      Call<GrpcInitService, grpc::InitService::Service,                   \
           method##Request, method##Response>::                           \
      EnqueueRequest(&service_, cq_.get(),                         \
                     &grpc::InitService::Service::Request##method,        \
                     &GrpcInitService::method##Handler);                  \
  } while (0)


    void HandleRPCsLoop() {
      ENQUEUE_REQUEST(ExchangeMachineInfo);
      void* tag;
      bool ok;
      while(cq_->Next(&tag, &ok)) {
        UntypedCall<GrpcInitService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcInitService>::Tag*>(tag);
        if(callback_tag) callback_tag->OnCompleted(this);
      }//while
    }

    std::unique_ptr<::grpc::ServerCompletionQueue> cq_;

    grpc::InitService::Service service_;

  private:
    template <class RequestMessage, class ResponseMessage>
    using ServiceCall = Call<GrpcInitService, grpc::InitService::Service,
                             RequestMessage, ResponseMessage>;

    void ExchangeMachineInfoHandler(ServiceCall<ExchangeMachineInfoRequest,
                                                ExchangeMachineInfoResponse>* call) {
      //TODO[xiaoshu]
    }

    void ExchangeMemoryDesc(ServiceCall<ExchangeMemoryDescRequest, 
                                        ExchangeMemoryDescResponse>* call) {
      //TODO[xiaoshu]
    }
    
};

}

#endif /* !GRPC_SERVER_SERVICE_H */
