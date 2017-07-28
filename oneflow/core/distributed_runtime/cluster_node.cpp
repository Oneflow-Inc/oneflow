#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unordered_map>

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"

DEFINE_string(server_def_filepath, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  ServerDef server_def;
  ParseProtoFromTextFile(FLAGS_server_def_filepath, &server_def);

  std::string this_node_name = server_def.this_node_name();
  LOG(INFO) << "Starting Up Node: " << this_node_name;

  std::unique_ptr<ServerInterface> grpc_server;
  GrpcServer::Create(server_def, &grpc_server);
  grpc_server->Start();
  return 0;
}
