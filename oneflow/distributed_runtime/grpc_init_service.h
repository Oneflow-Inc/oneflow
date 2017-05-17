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

namespace oneflow {
 
using ::grpc::ServerBuilder;

class GrpcInitService {
  public:
    GrpcInitService(::grpc::ServerBuilder* builder);
    ~GrpcInitService();

    void HandleRPCsLoop();

    std::unique_ptr<::grpc::ServerCompletionQueue> cq_;

    oneflow::InitService::AsyncService service_;

  private:
    template <class RequestMessage, class ResponseMessage>
    using ServiceCall = Call<GrpcInitService, grpc::InitService::AsyncService,
                             RequestMessage, ResponseMessage>;

    void ExchangeMachineInfoHandler(ServiceCall<ExchangeMachineInfoRequest,
                                                ExchangeMachineInfoReponse>* call) {
      //TODO[xiaoshu]
    }

    void ExchangeMemoryDesc(ServiceCall<ExchangeMemoryDescRequest,
                                        ExchangeMemoryDescReponse>* call) {
      //TODO[xiaoshu]
    }
    
};

}

#endif /* !GRPC_SERVER_SERVICE_H */
