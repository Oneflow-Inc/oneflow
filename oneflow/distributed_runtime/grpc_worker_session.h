/*
 * grpc_worker_session.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_WORKER_SESSION_H
#define GRPC_WORKER_SESSION_H

#include "distributed_runtime/grpc_session.h"

namespace oneflow {

class GrpcWorkerSession : public GrpcSession {
  public:
    GrpcWorkerSession();
    ~GrpcWorkerSession();
    void GetMachineDesc();
    void GetMemoryDesc();
    //void SendMessage();
    //void ReadData();
};


}

#endif /* !GRPC_WORKER_SESSION_H */
