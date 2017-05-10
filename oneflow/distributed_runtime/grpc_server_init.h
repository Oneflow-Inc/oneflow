/*
 * grpc_server_init.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_SERVER_INIT_H
#define GRPC_SERVER_INIT_H

#include "distributed_runtime/topology.pb.h"

namespace oneflow {

class GrpcServer {
  public:
    GrpcServer();
    ~GrpcServer();

    void Init_topology(Topology& topology);
    void Start_service();
    void Init_Context();

  private:


};

}

#endif /* !GRPC_SERVER_INIT_H */
