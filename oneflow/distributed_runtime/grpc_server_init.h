/*
 * grpc_server_init.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_SERVER_INIT_H
#define GRPC_SERVER_INIT_H

#include <grpc++/grpc++.h>

#include "distributed_runtime/topology.pb.h"
#include "distributed_runtime/machine_list.pb.h"

namespace oneflow {

using ::grpc::Channel;

class GrpcServer {
  public:
    GrpcServer();
    ~GrpcServer();

    void InitTopology(Topology topology, std::string& TopoFilePath, 
        MachineList machine, std::string& ResourceFilePath);
    void CreateChannelCache();
    void StartService();
    void InitContext();

    std::vector<std::string> vec_;
    std::map<std::string, std::string> pair_map_;

    struct machine_desc {
      std::string id;
      std::string name;
      std::string ip;
      std::string port;
    };
    std::map<std::string, machine_desc> machine_list_;

    std::map<std::string, ::grpc::Channel> channel_map_;

};

}

#endif /* !GRPC_SERVER_INIT_H */
