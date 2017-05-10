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

    void InitTopology(Topology topology, std::string& FilePath);
    void StartService();
    void InitContext();

    std::vector<std::string> vec_;

};

}

#endif /* !GRPC_SERVER_INIT_H */
