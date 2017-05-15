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

    oneflow::InitService::AsyncService init_service_;
    
};

}

#endif /* !GRPC_SERVER_SERVICE_H */
