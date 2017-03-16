#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>
#include "worker_service.h"
#include "oneflow.pb.h"
#include "oneflow.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using of::Role;
using of::Rank;
using of::Status_all;
using of::comm;

int main(int argc, char** argv){
    grpc::CreateChannel("10.120.15.3:50061", grpc::InsecureChannelCredentials());
    //std::string role_name("worker::10.120.15.5:50061");

    return 0;
}
