#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "lr.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using lr::Role;
using lr::Rank;
using lr::Status_all;
using lr::comm;

class lrServiceImpl final : public comm::Service{
    
    Status role_register(ServerContext* context, const Role* req, 
        Status_all* rep) override{
      std::string role_name = req->role_name();
      if(rank_map.find(role_name) == rank_map.end()){
          rank_map.insert(std::pair<std::string, int>(role_name, rank));
          ++rank;
      }
      rep->set_status("OK!");
      return Status::OK;
    }
  
    Status get_rank(ServerContext* context, const Role* req,
        Rank* rep) override{
        std::string role_name = req->role_name();
        int rank_id = rank_map[role_name];
        rep->set_rank(rank_id);
        return Status::OK;
    }

  private:
    std::map<std::string, int> rank_map;
    int rank = 0;
};

void RunServer(){
    std::string server_address("0.0.0.0:50061");
    lrServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout<<"Server listening on "<<server_address<<std::endl;
    server->Wait();
}

int main(int argc, char** argv){
  RunServer();
  return 0;
}
