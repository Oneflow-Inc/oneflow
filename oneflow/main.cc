#include <iostream>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "grpc_server_lib.h"


int main(){
  /*
  ServerDef server_def;
  {
    std::ifstream confFile("tfcluster.txt");
    google::protobuf::io::IstreamInputStream in(&confFile);
    if(!google::protobuf::TextFormat::Parse(&in, &server_def))
      confFile.close();
  }
  */
  oneflow::GrpcServer* server;
  server->NewServer();  
  
  return 0;
}
