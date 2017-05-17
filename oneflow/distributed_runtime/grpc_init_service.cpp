/*
 * grpc_server_service.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_init_service.h"

namespace oneflow {

GrpcInitService::GrpcInitService(::grpc::ServerBuilder* builder) {
  builder->RegisterService(&service_);
  cq_ = builder->AddCompletionQueue();
}

GrpcInitService::~GrpcInitService() {}

#define ENQUEUE_REQUEST(method, supports_cancel)                          \
  do {                                                                    \
    Call<GrpcInitService, grpc::InitService::Service,                     \
      method##Request, method##Response>::                                \
        EnqueueRequest(&master_service_, cq_.get(),                       \
                       &grpc::InitService::Service::Request##method,      \
                       &GrpcInitService::method##Handler);                \
  } while (0)

void GrpcInitService::HandleRPCsLoop() {

  void* tag;
  bool ok;
  while(cq_->Next(&tag, &ok)) {
    UntypedCall<GrpcInitService>::Tag* callback_tag =
      static_cast<UntypedCall<GrpcInitService>::Tag*>(tag);
    if(callback_tag) callback_tag->OnCompleted(this);
  }//while
}//HandleRPCsLoop

}



