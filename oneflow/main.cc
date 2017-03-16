#include <iostream>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "grpc_server_lib.h"

using tensorflow::ClusterDef;
using tensorflow::JobDef;
using tensorflow::ServerDef;

int main(){
  ServerDef server_def;
  {
    std::ifstream confFile("tfcluster.txt");
    google::protobuf::io::IstreamInputStream in(&confFile);
    if(!google::protobuf::TextFormat::Parse(&in, &server_def))
    confFile.close();
  }
  
  
  /*
  int length = server->ByteSize();
  char* buf = new char[length];
  server->SerializeToArray(buf, length);

  //parse
  ServerDef serverparse;
  serverparse.ParseFromArray(buf, length);
  std::cout<<serverparse.job_name()<<std::endl;
  std::cout<<serverparse.task_index()<<std::endl;
 
  auto pcluster = serverparse.cluster();
  auto pjob = pcluster.mutable_job();
  for(auto iter = pjob->begin(); iter != pjob->end(); ++iter){
    std::cout<<iter->name()<<" ";
    for(auto task : iter->tasks()){
      std::cout<<task.first<<" "<<task.second<<std::endl;
    }
  }
  */
  std::unique_ptr<tensorflow::ServerInterface> serverinterface;
  
  tensorflow::NewServer(server_def, &serverinterface);
  return 0;
}
