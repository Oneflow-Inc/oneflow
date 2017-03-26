#include <iostream>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>


#include <glog/logging.h>

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include "proto/oneflow.pb.h"
#include "proto/proto_io.h"
#include "context/one.h"
#include "context/machine_descriptor.h"
#include "context/id_map.h"
#include "memory/blob.h"


//#include "server_lib.h"
#include "net/grpc_server_lib.h"
//#include "public_session.h"


DEFINE_string(solver, "",
  "The solver definition protocol buffer text file.");

//void Session_test(const Options* opts) {
  

//}


int main(int argc, char* argv[]){
  /*
  ServerDef server_def;
  {
    std::ifstream confFile("tfcluster.txt");
    google::protobuf::io::IstreamInputStream in(&confFile);
    if(!google::protobuf::TextFormat::Parse(&in, &server_def))
      confFile.close();
  }
  */
  std::unique_ptr<oneflow::ServerInterface> server;
  //oneflow::NewServer(&server);
  oneflow::GrpcServer::Create(&server);
  server->Start();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  using Dtype = float;
  oneflow::SolverProto solver_param;
  oneflow::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  
  oneflow::TheOne<Dtype>::InitResource(FLAGS_solver);
  oneflow::TheOne<Dtype>::InitJob2(solver_param); 
  return 0;
}

