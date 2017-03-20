#include <iostream>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
/*
#include <glog/logging.h>

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include "caffe.pb.h"
#include "proto_io.h"
#include "context/one.h"
#include "context/machine_descriptor.h"
#include "context/id_map.h"
#include "memory/blob.h"
*/
#include "grpc_server_lib.h"

//DEFINE_string(solver, "",
//  "The solver definition protocol buffer text file.");

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
  oneflow::GrpcServer* server;
  server->NewServer();  
  /*
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  using Dtype = float;
  caffe::SolverProto solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  caffe::TheOne<Dtype>::InitResource(FLAGS_solver);
  caffe::TheOne<Dtype>::InitJob2(solver_param); 
  */
  return 0;
}
