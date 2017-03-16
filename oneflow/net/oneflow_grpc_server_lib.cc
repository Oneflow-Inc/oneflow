#include <iostream>
#include <memory>
#include <string>
#include <grpc++/grpc++.h>
#include "./protos/oneflow.grpc.pb.h"
//master services
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
//worker serviecs
using grpc::Channel;
using grpc::ClientContxt;

using oneflow::Role;
using oneflow::Rank;
using oneflow::Status_all;
using oneflow::comm;

GrpcServer::GrpcServer(){}
GrpcServer::~GrpcServer(){}

int GrpcServer::Init(){

}
