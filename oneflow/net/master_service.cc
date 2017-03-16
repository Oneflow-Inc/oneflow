#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>
#include "master_service.h"
#include "oneflow.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using of::Role;
using of::Rank;
using of::Status_all;
using of::comm;

void RunServer(){
    std::string server_address("0.0.0.0:50061");
    //lrServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    //builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout<<"Server listening on "<<server_address<<std::endl;
    server->Wait();
}

