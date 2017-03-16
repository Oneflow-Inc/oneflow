#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "./protos/lr.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using lr::Role;
using lr::Rank;
using lr::Status_all;
using lr::comm;

class lrClient {
    public:
      lrClient(std::shared_ptr<Channel> channel) : stub_(comm::NewStub(channel)) {}

      std::string of(const std::string& role_name){
          Role req_role;
          req_role.set_role_name(role_name);
          req_role.set_ip("10.120.15.5");
          
          Status_all rep_status;
          ClientContext context;
          Status status = stub_->role_register(&context, req_role, &rep_status);
          
          if(status.ok()){
              return rep_status.status(); 
          } else {
              std::cout<<status.error_code() << ": " << status.error_message()<< std::endl;
              return "RPC failed";
          }
      }

      int get_rank(std::string role_name){
          Role req;
          req.set_role_name(role_name);

          Rank rank;
          ClientContext context;
          Status status = stub_->get_rank(&context, req, &rank);
          
          if(status.ok()){
              return rank.rank();
          } else {
              std::cout<<status.error_code() <<":" << status.error_message()<<std::endl;
              return -1;
          }
      }

    private:
      std::unique_ptr<comm::Stub> stub_;
};

int main(int argc, char** argv){
    lrClient lr(grpc::CreateChannel("10.120.15.3:50061", grpc::InsecureChannelCredentials()));
    //std::string role_name("worker::10.120.15.5:50061");
    std::string role_name(argv[1]);
    std::string reply = lr.of(role_name);
    std::cout<<"register status: "<<reply<<std::endl;

    int rank_id = lr.get_rank(role_name);
    std::cout<<"my rank is "<<rank_id<<std::endl;
    return 0;
}
